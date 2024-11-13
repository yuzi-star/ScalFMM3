// --------------------------------
// See LICENCE file at project root
// File : algorithm/omp/upward.hpp
// --------------------------------
#ifndef SCALFMM_ALGORITHMS_MPI_UPWARD_HPP
#define SCALFMM_ALGORITHMS_MPI_UPWARD_HPP

#ifdef _OPENMP

#include <omp.h>

#include "scalfmm/algorithms/omp/upward.hpp"
// #include "scalfmm/operators/m2m.hpp"
// #include "scalfmm/operators/tags.hpp"
// #include "scalfmm/tree/utils.hpp"
// #include "scalfmm/utils/massert.hpp"
// #include "scalfmm/utils/math.hpp"

#include <cpp_tools/parallel_manager/parallel_manager.hpp>

namespace scalfmm::algorithms::mpi::pass
{
    /**
     * @brief Perform the communications for the children level
     *
     * @tparam Tree
     * @tparam Approximation
     */
    template<typename Tree>
    inline auto start_communications(const int& level, Tree& tree) -> void
    {
        using value_type = typename Tree::base_type::cell_type::value_type;
        using dep_type = typename Tree::group_of_cell_type::symbolics_type::ptr_multi_dependency_type;

        static constexpr std::size_t dimension = Tree::base_type::box_type::dimension;
        static constexpr int nb_inputs = Tree::cell_type::storage_type::inputs_size;
        //
        // number of theoretical children
        constexpr int nb_children = math::pow(2, dimension);
        static constexpr auto prio{omp::priorities::max};
        //
        auto& para = tree.get_parallel_manager();
        auto comm = para.get_communicator();
        auto rank = comm.rank();
        int nb_proc = comm.size();
        int tag_nb = 1200 + 10 * level;
        int tag_data = 1201 + 10 * level;
        auto mpi_int_type = cpp_tools::parallel_manager::mpi::get_datatype<int>();

        auto level_child = level + 1;
        // get size of multipoles
        auto it_group1 = tree.begin_mine_cells(level_child);
        auto it_cell1 = it_group1->get()->begin();
        auto const& m1 = it_cell1->cmultipoles();
        int size{int(nb_inputs * m1.at(0).size()) * nb_children};

        std::vector<dep_type> deps;

        if(rank > 0)
        {
            // If we send the multipoles, they must have been updated by the M2M of the previous level!

            int count{0};
            // serialization
            std::vector<value_type> buffer(size);
            // check if we have to send some children
            auto first_group_of_child = tree.begin_mine_cells(level_child)->get();
            auto last_index_child = first_group_of_child->component(0).index();
            //
            auto first_group = tree.begin_mine_cells(level)->get();
            auto index = first_group->component(0).index();
            // Check if I have the parent
            // std::cout << " upward " << index << "  parent of last child index" << (last_index_child >> dimension)
            //           << "comm " << std::boolalpha << (index > last_index_child >> dimension) << std::endl;
            // std::cout << " send comm to " << rank - 1 << std::endl;

            // Should be in a task !
            auto it_group = tree.begin_mine_cells(level_child);
            auto gs = it_group->get()->size();
            int nb_grp_dep = std::min(static_cast<int>(nb_children / gs + 1),
                                      static_cast<int>(std::distance(it_group, tree.end_mine_cells(level_child))));
            auto it_grp = it_group;
#ifdef M2M_COMM_TASKS
            for(int i = 0; i < nb_grp_dep; ++i, ++it_grp)
            {
                deps.push_back(&(it_grp->get()->ccomponent(0).cmultipoles(0)));
            }
#pragma omp task shared(rank, mpi_int_type, size, tag_data, tag_nb, dimension) firstprivate(it_group)                  \
  depend(iterator(std::size_t it = 0                                                                                   \
                  : deps.size()),                                                                                      \
         out                                                                                                           \
         : (deps[it])[0]) priority(prio)
	    #endif
            {
                if(index > last_index_child >> dimension)
                {
                    // index is now the parent of the first child
                    index = last_index_child >> dimension;
                    // I have to send
                    // find the number of children to send (get pointer on multipoles !!)
                    // Construct an MPI datatype

                    // serialization

                    // auto it_group = tree.begin_mine_cells(level_child);
                    auto it_cell = it_group->get()->begin();
                    auto next = scalfmm::component::generate_linear_iterator(1, it_group, it_cell);
                    // std::vector<> cells_to_send();
                    //  We construct an MPI DATA_TYPE
                    // MPI_Datatype mult_data_type;
                    auto it = std::begin(buffer);
                    for(int i = 0; i < nb_children - 1; ++i, next())
                    {
                        // std::cout << " Check children  P " << index << " C " << it_cell->index() << std::endl;
                        if(index < (it_cell->index() >> dimension))
                        {
                            break;
                        }
                        // copy the multipoles in the buffer
                        auto const& m = it_cell->cmultipoles();

                        auto nb_m = m.size();
                        // std::cout << "cell index: " << it_cell->index() << " level " << it_cell->csymbolics().level
                        // <<
                        // "\n";
                        for(std::size_t i{0}; i < nb_m; ++i)
                        {
                            auto const& ten = m.at(i);
                            std::copy(std::begin(ten), std::end(ten), it);
                            it += ten.size();
                        }
                        ++count;
                    }
                }

                comm.isend(&count, 1, mpi_int_type, rank - 1, tag_nb);

                // std::cout << "nb_send = " << count << std::endl;

                if(count != 0)
                {
                    // loop to serialize the multipoles
                    auto mpi_type = cpp_tools::parallel_manager::mpi::get_datatype<value_type>();

                    comm.isend(buffer.data(), size, mpi_type, rank - 1, tag_data);
                    // std::cout << "buffer:";
                    // for(int i = 0; i < 10; ++i)
                    // {
                    //     std::cout << "  " << buffer[i];
                    // }
                }
            }
        }
        if(rank == nb_proc - 1)
        {
            return;
        }
        // Add task dependencies
        //
        // We receive the cells (at most 2^d - 1) to have all the children of
        //  the last father cell I own. These cells go into the first phantom group on the right.
        // dep(out) these cells
        // dep(out) group_parent_dep[0]??

        auto it_group = tree.end_mine_cells(level_child);
        auto gs = it_group->get()->size();
        int nb_grp_dep = std::min(static_cast<int>(nb_children / gs + 1),
                                  static_cast<int>(std::distance(it_group, tree.end_cells(level_child))));
        auto it_group_parent = --(tree.end_mine_cells(level));
        auto it_grp = it_group;
        for(int i = 0; i < nb_grp_dep; ++i, ++it_grp)
        {
            deps.push_back(&(it_grp->get()->ccomponent(0).cmultipoles(0)));
        }

        auto group_parent_dep = it_group_parent->get()->ccomponent(0).cmultipoles(0);
#ifdef M2M_COMM_TASKS
#pragma omp task shared(rank, mpi_int_type, size, tag_data, tag_nb) depend(iterator(std::size_t it = 0                 \
                                                                                    : deps.size()),                    \
                                                                           out                                         \
                                                                           : (deps[it])[0]) priority(prio)
	#endif
        {
            int count{-1};
            comm.recv(&count, 1, mpi_int_type, rank + 1, tag_nb);
            // std::cout << "     We receive " << count << " cell(s)\n";
            // use a recv
            if(count > 0)
            {
                std::vector<value_type> buffer(size);
                auto mpi_type = cpp_tools::parallel_manager::mpi::get_datatype<value_type>();
                comm.recv(buffer.data(), size, mpi_type, rank + 1, tag_data);
                //
                // std::cout << "buffer:";
                // for(int i = 0; i < 10; ++i)
                // {
                //     std::cout << "  " << buffer[i];
                // }
                std::cout << std::endl;
                auto it_group = tree.end_mine_cells(level_child);
                auto it_cell = it_group->get()->begin();
                auto next1 = scalfmm::component::generate_linear_iterator(1, it_group, it_cell);
                auto it = std::begin(buffer);

                for(int i = 0; i < count; ++i, next1())
                {
                    // copy the multipoles in the buffer
                    auto& m = it_cell->multipoles();
                    //   auto tm = cell.ctransformed_multipoles();

                    auto nb_m = m.size();
                    // std::cout << "cell index: " << it_cell->index() << " level " << it_cell->csymbolics().level <<
                    // "\n";
                    for(std::size_t i{0}; i < nb_m; ++i)
                    {
                        auto& ten = m.at(i);
                        std::copy(it, it + ten.size(), std::begin(ten));
                        it += ten.size();
                    }
                }
            }
        }
        // std::cout << " END start_communications" << std::endl << std::flush;
        //
    }

    /// @brief This function constructs the local approximation for all the cells of the tree by applying the operator
    /// m2m
    ///
    /// @param tree   the tree target
    /// @param approximation the approximation to construct the local approximation
    ///
    template<typename Tree, typename Approximation>
    inline auto upward(Tree& tree, Approximation const& approximation) -> void
    {
        auto leaf_level = tree.height() - 1;

        // upper working level is
        const int top_height = tree.box().is_periodic() ? 0 : 2;
        // const int start_duplicated_level = tree.start_duplicated_level();
        //
        // int top = start_duplicated_level < 0 ? top_height : start_duplicated_level - 1;
        int top = top_height;
        for(int level = leaf_level - 1; level >= top /*top_height*/; --level)   // int because top_height could be 0
        {
            // std::cout << "M2M : " << level + 1 << " -> " << level << std::endl << std::flush;
            //
            start_communications(level, tree);
            // std::cout << "  end comm " << std::endl << std::flush;

            omp::pass::upward_level(level, tree, approximation);
            // std::cout << "  end upward_level " << level << std::endl << std::flush;
        }
        // std::cout << "end upward " << std::endl << std::flush;

        //

        // for(int level = start_duplicated_level; level >= top_height; --level)   // int because top_height could be 0
        // {
        //     std::cout << "Level duplicated (seq): " << level << std::endl;
        //     upward_level(level, tree, approximation);
        // }
    }
}   // namespace scalfmm::algorithms::mpi::pass

#endif   // _OPENMP
#endif   // SCALFMM_ALGORITHMS_MPI_UPWARD_HPP
