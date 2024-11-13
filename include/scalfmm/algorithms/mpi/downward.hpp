// --------------------------------
// See LICENCE file at project root
// File : algorithm/mpi/downward.hpp
// --------------------------------
#ifndef SCALFMM_ALGORITHMS_MPI_DOWNWARD_HPP
#define SCALFMM_ALGORITHMS_MPI_DOWNWARD_HPP

#include "scalfmm/operators/l2l.hpp"
#ifdef _OPENMP

#include <omp.h>

#include "scalfmm/algorithms/omp/downward.hpp"
#endif   // _OPENMP

#include <cpp_tools/parallel_manager/parallel_manager.hpp>

namespace scalfmm::algorithms::mpi::pass
{
    /**
     * @brief perform the l2l communications for the father level
     *
     *
     * @tparam Tree
     * @tparam Approximation
     * @param level current father level
     * @param tree the tree
     */
    template<typename Tree>
    inline auto downward_communications_level(const int& level, Tree& tree) -> void
    {
        using value_type = typename Tree::base_type::cell_type::value_type;
        static constexpr int nb_inputs = Tree::cell_type::storage_type::inputs_size;
        static constexpr std::size_t dimension = Tree::base_type::box_type::dimension;

        auto child_level = level + 1;
        auto const& distrib = tree.get_cell_distribution(child_level);

        // compute the size of the multipoles to send (generic) versus  math::pow(order, dimension)
        auto it_group = tree.end_mine_cells(level) - 1;       // last group the I own
        auto pos = it_group->get()->size() - 1;               // index of the last cell in the group
        auto const& cell = it_group->get()->component(pos);   // the cell

        auto const& m = cell.cmultipoles();
        auto size{int(nb_inputs * m.at(0).size())};
        // For the communications
        auto& para = tree.get_parallel_manager();
        auto comm = para.get_communicator();
        auto rank = comm.rank();
        int nb_proc = comm.size();
        int tag_data = 2201 + 10 * level;

        // Send
        if(rank != nb_proc - 1)
        {
            auto last_child_index = distrib[rank][1] - 1;
            auto first_child_index_after_me = distrib[rank + 1][0];
            // dependencies in on th group
            if((last_child_index >> dimension) == (first_child_index_after_me >> dimension))
            {
                std::vector<value_type> buffer(size);

                // I have to send a message from my right to update the multipoles of the first
                // cells of the right ghosts.
                // temporary buffer

                auto nb_m = m.size();
                // std::cout << "cell index: " << cell.index() << " = parent " << (last_child_index >> dimension) <<
                // "\n"; loop to serialize the multipoles
                auto it = std::begin(buffer);
                for(std::size_t i{0}; i < nb_m; ++i)
                {
                    auto const& ten = m.at(i);
                    std::copy(std::begin(ten), std::end(ten), it);
                    it += ten.size();
                }
                auto mpi_type = cpp_tools::parallel_manager::mpi::get_datatype<value_type>();

                comm.isend(buffer.data(), size, mpi_type, rank + 1, tag_data);
            }
        }
        // Receive
        if(rank > 0)
        {
            auto last_child_index_before_me = distrib[rank - 1][1] - 1;
            auto first_child_index = distrib[rank][0];
            // dependencies out on the group

            if((last_child_index_before_me >> dimension) == (first_child_index >> dimension))
            {
                std::vector<value_type> buffer(size);

                // I have to receive a message from my left to update the multipoles of the last
                // cells of the left ghosts.
                auto mpi_type = cpp_tools::parallel_manager::mpi::get_datatype<value_type>();
                comm.recv(buffer.data(), size, mpi_type, rank - 1, tag_data);

                /// set the multipoles in the ghost
                auto it_group = tree.begin_mine_cells(level) - 1;   // last left ghosts
                auto pos = it_group->get()->size() - 1;             // index of the last cell in the group
                auto& cell = it_group->get()->component(pos);
                auto& m = cell.multipoles();
                //   auto tm = cell.ctransformed_multipoles();
                // std::cout << "cell index: " << cell.index() << " = parent " << (last_child_index_before_me >>
                // dimension)
                //           << "\n";

                auto nb_m = m.size();
                // std::cout << "cell index: " << it_cell->index() << " level " << it_cell->csymbolics().level <<
                // "\n";
                auto it = std::begin(buffer);
                for(std::size_t i{0}; i < nb_m; ++i)
                {
                    auto& ten = m.at(i);
                    std::copy(it, it + ten.size(), std::begin(ten));
                    it += ten.size();
                }
            }
        }
    }
    /// @brief This function constructs the local approximation for all the cells of the tree by applying the
    /// operator l2l
    ///
    /// @param tree   the tree target
    /// @param approximation the approximation to construct the local approximation
    ///
    template<typename Tree, typename Approximation>
    inline auto downward(Tree& tree, Approximation const& approximation) -> void
    {
        // upper working level is
        const auto top_height = tree.box().is_periodic() ? 0 : 2;
        const auto leaf_level = tree.leaf_level();
        // std::cout << "start downward pass top_height " << top_height << " leaf_level " << leaf_level << std::endl
        //           << std::flush;

        for(std::size_t level = top_height; level < leaf_level; ++level)
        {
            // std::cout << "downward  : " << level << std::endl << std::flush;
            // std::cout << "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&" << std::endl;
            downward_communications_level(level, tree);
            // std::cout << "   end downward comm  " << level << std::endl << std::flush;
            // std::cout << "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&" << std::endl;

            omp::pass::downward_level(level, tree, approximation);
        }
        // std::cout << "end downward pass" << std::endl << std::flush;
    }
}   // namespace scalfmm::algorithms::mpi::pass

#endif   // SCALFMM_ALGORITHMS_MPI_DOWNWARD_HPP
