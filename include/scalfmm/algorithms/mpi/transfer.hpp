// --------------------------------
// See LICENCE file at project root
// File : algorithm/omp/upward.hpp
// --------------------------------
#ifndef SCALFMM_ALGORITHMS_MPI_TRANSFER_HPP
#define SCALFMM_ALGORITHMS_MPI_TRANSFER_HPP

#ifdef _OPENMP

#include <map>
#include <omp.h>
#include <utility>

#include "scalfmm/algorithms/omp/transfer.hpp"
#include "scalfmm/meta/traits.hpp"
#include "scalfmm/operators/tags.hpp"
#include "scalfmm/parallel/mpi/comm.hpp"
#include "scalfmm/parallel/mpi/utils.hpp"
#include "scalfmm/parallel/utils.hpp"
#include "scalfmm/tree/utils.hpp"
#include "scalfmm/utils/massert.hpp"
#include "scalfmm/utils/math.hpp"

#include <cpp_tools/parallel_manager/parallel_manager.hpp>

namespace scalfmm::algorithms::mpi::pass
{

    /**
     * @brief Perform the communications between the tree_source and the tree_target for the current level
     *
     *  The algorithm is done in three steps
     *   step 1 find the number of data to exchange
     *   step 2 construct the morton index to send
     *   step 3 send the multipoles
     *
     * @tparam TreeS
     * @tparam TreeT
     * @param level  level in the tree
     * @param tree_source  source tree (contains the multipoles)
     * @param tree_target  target tree
     */
    template<typename TreeS, typename TreeT>
    inline auto start_communications(const int& level, TreeS& tree_source, TreeT& tree_target) -> void
    {
        using mortonIdx_type = std::int64_t;   // typename TreeT::group_of_cell_type::symbolics_type::morton_type;
        static constexpr int nb_inputs = TreeS::base_type::cell_type::storage_type::inputs_size;
        static constexpr int dimension = TreeS::base_type::box_type::dimension;

        using value_type_ori = typename TreeS::base_type::cell_type::storage_type::transfer_multipole_type;
        using value_type1 = value_type_ori;
        using value_type = std::conditional_t<meta::is_complex_v<value_type_ori>,
                                              meta::has_value_type_t<value_type_ori>, value_type_ori>;

        int nb_values = meta::is_complex_v<value_type_ori> ? 2 : 1;
	//         std::cout << " Complex " << std::boolalpha << meta::is_complex_v<value_type_ori> << " nb val=" << nb_values
	//                   << "  value_type " << typeid(value_type).name() << "  value_type_ori "
	//                   << typeid(value_type_ori).name() << std::endl;
        auto& para = tree_target.get_parallel_manager();
        auto comm = para.get_communicator();

        auto rank = para.get_process_id();
        auto nb_proc = para.get_num_processes();
        if(nb_proc == 1)
        {   // Openmp case -> no communication
            return;
        }
        std::vector<int> nb_messages_to_send(nb_proc, 0);
        std::vector<int> nb_messages_to_receive(nb_proc, 0);

        ///////////////////////////////////////////////////////////////////////////////////
        /// STEP 1
        ///////////////////////////////////////////////////////////////////////////////////
        /// Determines the Morton index vector to be received from processor p? In addition, for each Morton index we
        /// store the cell, i.e. a pointer to its group and the index within the group (group_ptr, index). This will
        /// enable us to insert the multipoles received from processor p directly into the cell.
        ///
        /// to_receive: a vector of vector of pair (the iterator on the group ,the position of the cell in the group)
        ///  to_receive[p] is the position vector of cells in groups whose Morton index comes from processor p
        ///  to_receive[p][i] a pair (the iterator on the group ,the position of the cell in the group)
        /// vector of size nb_proc
        ///    - nb_messages_to_receive: the number of morton indices to  exchange with processor p
        ///    - nb_messages_to_send: the number of morton indices to  send tp processor p
        ///    - morton_to_recv: the morton indices to  exchange with processor p
        /////////////////////////////////////////////////////////////////////////////////
        using grp_access_type = std::pair<decltype(tree_target.begin_cells(level)), int>;
        std::vector<std::vector<grp_access_type>> to_receive(nb_proc);
        std::vector<std::vector<mortonIdx_type>> morton_to_receive(nb_proc);   // TOREMOVE
                                                                               // #ifdef SPLIT_COMM
        {
            auto begin_left_ghost = tree_target.begin_cells(level);

            auto end_left_ghost = tree_target.begin_mine_cells(level);
            auto begin_right_ghost = tree_target.end_mine_cells(level);
            auto end_right_ghost = tree_target.end_cells(level);
            auto const& distrib = tree_source.get_cell_distribution(level);
            //
            scalfmm::parallel::comm::start_step1(comm, begin_left_ghost, end_left_ghost, begin_right_ghost,
                                                 end_right_ghost, distrib, nb_messages_to_receive, nb_messages_to_send,
                                                 to_receive, morton_to_receive);
        }
        //         for(auto p = 0; p < nb_proc; ++p)
        //         {
        //             io::print("    morton to receive[" + std::to_string(p) + "] ", morton_to_receive[p]);
        //         }
        //         io::print("    nb_messages_to_receive ", nb_messages_to_receive);
        //         io::print("    nb_messages_to_send ", nb_messages_to_send);

        ///////////////////////////////////////////////////////////////////////////////////
        /// STEP 2
        ///
        // We can now exchange the morton indices
        // Morton's list of indices to send their multipole to proc p
        std::vector<std::vector<mortonIdx_type>> morton_to_send(nb_proc);
        std::vector<cpp_tools::parallel_manager::mpi::request> tab_mpi_status;

        scalfmm::parallel::comm::start_step2(nb_proc, rank, comm, nb_messages_to_receive, nb_messages_to_send,
                                             morton_to_receive, morton_to_send);
	

	/*
         for(auto p = 0; p < nb_proc; ++p)
         {
             io::print("    morton_to_receive[" + std::to_string(p) + "] ", morton_to_receive[p]);
         }
         for(auto p = 0; p < nb_proc; ++p)
         {
             io::print("    morton_to_send[" + std::to_string(p) + "] ", morton_to_send[p]);
         }
	*/
        static constexpr auto prio{omp::priorities::max};
        /////////////////////////////////////////////////////////////////////////////////
        /// STEP 3
        /////////////////////////////////////////////////////////////////////////////////
        /// Processor p sends multipoles to update the multipoles of ghost cells in other processors, so that the
        /// transfer phase can be performed independently (as in OpenMP).
        /// We construct a buffer to pack all the multipoles (first version).
        /// In a second version, to save memory we construct a MPI_TYPE
        /// \warning  We need to make sure that the multipoles are up to date, so we need to put a dependency on the
        /// task.
        ///
        /// First version
        /// This stage is divided into two parts. First, for all the processors I need to communicate with, we pack all
        /// the multipoles to be sent into a buffer and send the buffer. Then we post the receptions to obtain the
        /// multipoles we're going to put in our ghost cells.
        /////////////////////////////////////////////////////////////////////////////////
        // type of dependence
        using dep_type = typename TreeS::group_of_cell_type::symbolics_type::ptr_multi_dependency_type;

        auto mpi_multipole_type = cpp_tools::parallel_manager::mpi::get_datatype<value_type>();
	//        std::cout << "\n Start step 3\n\n";
        auto nb_cells{morton_to_send[0].size()};
        for(auto const& p: morton_to_send)
        {
            nb_cells = std::max(nb_cells, p.size());
        }
        int order = tree_source.order();
        //         nb_values = 2 if complex type otherwise 1;
        //   math::pow(order, dimension) only works with interpolation not generic !!!!
        int size_mult{int(nb_inputs * math::pow(order, dimension)) * nb_values};   // WRONG !!!!!!

        // allocate the buffer to store the multipoles
        std::vector<std::vector<value_type_ori>> buffer(nb_proc);
        {
            // method to construct the buffer of multipoles to send
            auto build_buffer = [](auto first_group_ghost, auto last_group_ghost,
                                   std::vector<mortonIdx_type> const& index_to_send,
                                   std::vector<value_type_ori>& buffer)
            {
		try
		  {
		    //		    std::cout <<  " -----      build_buffer   ----\n" <<std::flush;
		    //		    std::cout <<  " -----      buffer size " <<buffer.size() <<std::endl<<std::flush;
		    //		    std::cout <<  " -----     index_to_send size " <<index_to_send.size() <<std::endl<<std::flush;
		    int idx{0};
		    int max_idx = index_to_send.size();   // loop on the groups
		    auto it = std::begin(buffer);
		    for(auto grp_ptr = first_group_ghost; grp_ptr != last_group_ghost; ++grp_ptr)
		      {
			int start_grp{0};
			
			auto const& csymb = (*grp_ptr)->csymbolics();
			// iterate on the cells
			while(idx < max_idx and math::between(index_to_send[idx], csymb.starting_index, csymb.ending_index))
			  {   // find cell inside the group
			    int pos{-1};
			    for(int i = start_grp; i < (*grp_ptr)->size(); ++i)
			      {
				auto morton = (*grp_ptr)->component(i).csymbolics().morton_index;
				if(index_to_send[idx] == morton)
				  {
				    pos = i;
				    start_grp = i + 1;
				    // std::cout << "   pos = " << pos << std::endl;
				    break;
				  }
			      }
				//			     std::cout << " morton to find " << index_to_send[idx] << " cell found "
			    //			               << (*grp_ptr)->component(pos).csymbolics().morton_index << '\n';
			    auto const& cell = (*grp_ptr)->component(pos);
			    auto const& m = cell.transfer_multipoles();
			    auto nb_m = m.size();
			    //			    std::cout << "          nb_m" <<  m.size() <<std::endl;
			    for(std::size_t i{0}; i < nb_m; ++i)
			      {
				auto const& ten = m.at(i);
				std::copy(std::begin(ten), std::end(ten), it);
				it += ten.size();
			      }
			    ++idx;
			  }
		      }
		    //		                  std::cout <<  " -----      build_buffer   ----\n" <<std::flush;
		  }
		catch(std::out_of_range& e)
		  {
		    std::cout <<" error in buffer building !!!!!!!!!\n"; 
		    std::cout << e.what() << '\n'<<std::flush;
		  }
            };

            // vector of dependencies on the group for each MPI process
            std::vector<std::vector<dep_type>> deps_send(nb_proc);

            // loop on the processors to construct the buffer and to send it
            for(auto p = 0; p < nb_proc; ++p)
            {
	      //	      std::cout << "  loop to send " << p <<"   "<<  morton_to_send[p].size() <<std::endl;
                /// check if I have something to send
                if(p != rank and morton_to_send[p].size() > 0)
                {
#ifdef TRANSFERT_COMM_TASKS
                    /// We first construct the in dependencies to ensure that multipoles
                    /// are updated by the  previous pass.
                    parallel::utils::build_dependencies_from_morton_vector(tree_source.begin_mine_cells(level),
                                                                           tree_source.end_mine_cells(level),
                                                                           morton_to_send[p], deps_send[p]);
/// spawn a task for sending communication to process p

		    #pragma omp task shared(tree_source, morton_to_send, buffer, rank, nb_proc, deps_send, nb_messages_to_receive, \
                        size_mult, mpi_multipole_type) firstprivate(p, level) depend(iterator(std::size_t it = 0       \
                                                                                              : deps_send[p].size()),  \
                                                                                     in                                \
                                                                                     : ((deps_send[p])[it])[0])        \
  priority(prio)
#endif
                    {
                        buffer[p].resize(morton_to_send[p].size() * size_mult);
			//                         std::cout << " prepare to processor " << p << '\n';
			//                         io::print("    --> morton_to_send[" + std::to_string(p) + "] ", morton_to_send[p]);
			//                         std::cout << '\n';
			//			 std::cout << " nb cells " << std::distance(tree_source.begin_mine_cells(level), tree_source.end_mine_cells(level)) <<std::endl<<std::flush;
                        build_buffer(tree_source.begin_mine_cells(level), tree_source.end_mine_cells(level),
                                     morton_to_send[p], buffer[p]);
			//			std::cout << " end build buffer  "<< p << std::endl << std::flush;

                        // send buffer to processor p
			//                         std::cout << " post send to proc " << p << " size= " << buffer[p].size() << std::endl
			//                                  << std::flush;
			//			 io::print("  buffer  send ",buffer[p]);
                        comm.isend(reinterpret_cast<value_type*>(buffer[p].data()), buffer[p].size(),
                                   mpi_multipole_type, p, 611);
                    }
                }
            }
        }
        // Reception of the multipoles
        {
            // Reset the array of requests used in step 2
            tab_mpi_status.clear();
            // We first construct the out dependencies (all the ghost groups)
#ifdef TRANSFERT_COMM_TASKS

            // compute the dependencies
            auto size_dep{std::distance(tree_source.begin_cells(level), tree_source.begin_mine_cells(level)) +
                          std::distance(tree_source.end_mine_cells(level), tree_source.end_cells(level))};
            std::vector<dep_type> deps_recv(size_dep);
            {   // Find all dependencies - naive version
                int idx{0};
                for(auto it_grp = tree_source.begin_cells(level); it_grp != tree_source.begin_mine_cells(level);
                    ++it_grp, ++idx)
                {
                    deps_recv[idx] = &(it_grp->get()->ccomponent(0).cmultipoles(0));
                }
                for(auto it_grp = tree_source.end_mine_cells(level); it_grp != tree_source.end_cells(level);
                    ++it_grp, ++idx)
                {
                    deps_recv[idx] = &(it_grp->get()->ccomponent(0).cmultipoles(0));
                }
            }
            // std::cout << "size_dep= " << size_dep << " " << deps_recv.size() << std::endl;
// post the task on the reception
#pragma omp task shared(rank, nb_proc, nb_messages_to_receive, size_mult, mpi_multipole_type)	\
  depend(iterator(std::size_t it = 0                                                                                   \
                  : deps_recv.size()),                                                                                 \
         out                                                                                                           \
        : (deps_recv[it])[0]) priority(prio)
#endif
            {
                // post the receives
                int cc{0};
                std::vector<cpp_tools::parallel_manager::mpi::request> recept_mpi_status;
                std::vector<std::vector<value_type_ori>> buffer_rep(nb_proc);

                for(auto p = 0; p < nb_proc; ++p)
                {
                    if(p != rank and nb_messages_to_receive[p] != 0)
                    {
                        buffer_rep[p].resize(nb_messages_to_receive[p] * size_mult);
			//      std::cout << " post comm to proc " << p << "size= " << nb_messages_to_receive[p] * size_mult
                        //           << std::endl   << std::flush;

                        recept_mpi_status.push_back(
                          comm.irecv(buffer_rep[p].data(), buffer_rep[p].size(), mpi_multipole_type, p, 611));
                        ++cc;
                    }
                }
		// std::cout << " Post reception  nb=" << cc << "   " << recept_mpi_status.size() << '\n' << std::flush;

                // wait we receive all the communication

                if(recept_mpi_status.size() > 0)
                {
                    cpp_tools::parallel_manager::mpi::request::waitall(recept_mpi_status.size(),
                                                                       recept_mpi_status.data());
                }
		//   std::cout << "  end wait all\n" << std::flush;

                // put the multipoles inside the ghosts
                for(auto p = 0; p < nb_proc; ++p)
                {
                    if(p != rank and to_receive[p].size() > 0)
                    {
                        auto const& buffer = buffer_rep[p];
			//                         std::cout << " proc p= " << p << std::endl << std::flush;
			//			 io::print("   buffer ",buffer) ;
                        // ONLY WORKS IF SOURCE == TARGET
                        auto const& pairs = to_receive[p];
                        auto it = std::begin(buffer);

                        for(auto i = 0; i < int(pairs.size()); ++i)
                        {
                            auto& cell = pairs[i].first->get()->component(pairs[i].second);
                            auto& m = cell.transfer_multipoles();
                            auto nb_m = m.size();

                            for(std::size_t i{0}; i < nb_m; ++i)
                            {
                                auto& ten = m.at(i);
                                std::copy(it, it + ten.size(), std::begin(ten));
                                it += ten.size();
                            }
                        }
                    }
                }
            }   // end task
	    //	    std::cout << "end step 3 \n"<<std::flush;
        }       // end step3
       para.get_communicator().barrier();
    }           // end function start_communications
    ///////////////////////////////////////////////////////////////////////////////////

    /// @brief apply the transfer operator to construct the local approximation in tree_target
    ///
    /// @tparam TreeS template for the Tree source type
    /// @tparam TreeT template for the Tree target type
    /// @tparam FarField template for the far field type
    /// @tparam BufferPtr template for the type of pointer of the buffer
    /// @param tree_source the tree containing the source cells/leaves
    /// @param tree_target the tree containing the target cells/leaves
    /// @param far_field The far field operator
    /// @param buffers vector of buffers used by the far_field in the transfer pass (if needed)
    /// @param split the enum  (@see split_m2l) tp specify on which level we apply the transfer
    /// operator
    ///
    template<typename TreeS, typename TreeT, typename FarField, typename BufferPtr>
    inline auto transfer(TreeS& tree_source, TreeT& tree_target, FarField const& far_field,
                         std::vector<BufferPtr> const& buffers, omp::pass::split_m2l split = omp::pass::split_m2l::full)
      -> void
    {
        //
        auto tree_height = tree_target.height();

        /////////////////////////////////////////////////////////////////////////////////////////
        ///  Loop on the level from the level top_height to the leaf level (Top to Bottom)
        auto top_height = tree_target.box().is_periodic() ? 1 : 2;
        auto last_level = tree_height;

        switch(split)
        {
        case omp::pass::split_m2l::remove_leaf_level:
            last_level--;
            ;
            break;
        case omp::pass::split_m2l::leaf_level:
            top_height = (tree_height - 1);
            break;
        case omp::pass::split_m2l::full:
            break;
        }
        for(std::size_t level = top_height; level < last_level; ++level)
        {
	  //             std::cout << "transfer  : " << level << std::endl << std::flush;
            start_communications(level, tree_source, tree_target);
	    //             std::cout << "   end comm  " << level << std::endl << std::flush;
	    //             std::cout << "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&" << std::endl;

            omp::pass::transfer_level(level, tree_source, tree_target, far_field, buffers);
	    //             std::cout << "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&" << std::endl;
        }
	//         std::cout << "end transfer pass" << std::endl << std::flush;
    }

}   // namespace scalfmm::algorithms::mpi::pass
#endif   //_OPENMP
#endif   // SCALFMM_ALGORITHMS_MPI_TRANSFER_HPP
