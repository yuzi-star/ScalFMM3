﻿// --------------------------------
// See LICENCE file at project root
// File : algorithm/omp/downward.hpp
// --------------------------------
#ifndef SCALFMM_ALGORITHMS_OMP_DOWNWARD_HPP
#define SCALFMM_ALGORITHMS_OMP_DOWNWARD_HPP

#ifdef _OPENMP
#include <limits>

#include "scalfmm/operators/l2l.hpp"
#include "scalfmm/tree/utils.hpp"
#include "scalfmm/utils/massert.hpp"
#include "scalfmm/utils/math.hpp"

namespace scalfmm::algorithms::omp::pass
{
    /**
     * @brief perform the l2l operator for a given level
     *
     * @tparam Tree
     * @tparam Approximation
     * @param level current level to construct the m2m
     * @param tree the tree
     * @param approximation the approximation
     */
    template<typename Tree, typename Approximation>
    inline auto downward_level(const int& level, Tree& tree, Approximation const& approximation) -> void
    {
        using scalfmm::operators::l2l;
        using value_type = typename Approximation::value_type;

        // Get the index of the corresponding child-parent interpolator
        std::size_t level_interpolator_index =
          (approximation.cell_width_extension() < std::numeric_limits<value_type>::epsilon()) ? 2 : level;
        // // parent level
        // iterator on the groups of cells (current level)
        auto group_of_cell_begin = tree.begin_mine_cells(level);
        auto group_of_cell_end = tree.end_mine_cells(level);   //
        // iterator on the groups of cells (child level)
        auto group_of_child_cell_begin = tree.begin_cells(level + 1);
        auto group_of_child_cell_end = tree.end_cells(level + 1);

        auto start_range_dependencies{group_of_cell_begin};
        auto end_range_dependencies{group_of_cell_begin};

        static constexpr auto prio{priorities::l2l};

        while(group_of_child_cell_begin != group_of_child_cell_end)
        {
            using interpolator_type = typename std::decay_t<Approximation>;
            static constexpr auto dimension{interpolator_type::dimension};

            using ptr_parent_groups_type = std::decay_t<decltype(group_of_cell_begin->get())>;

            auto group_child = group_of_child_cell_begin->get();
            auto group_child_raw = &group_child->ccomponent(0).clocals(0);
            auto child_group_starting_morton_index = group_child->csymbolics().starting_index;
            auto child_group_ending_morton_index = group_child->csymbolics().ending_index;
            auto parent_starting_morton_index = child_group_starting_morton_index >> dimension;
            auto parent_ending_morton_index = ((child_group_ending_morton_index - 1) >> dimension) + 1;

            auto& parent_dependencies{(*group_of_child_cell_begin)->symbolics().group_dependencies_l2l_in};

            std::tie(start_range_dependencies, end_range_dependencies) = index::get_parent_group_range(
              parent_starting_morton_index, parent_ending_morton_index, start_range_dependencies, group_of_cell_end);
            auto start_range_dependencies_tmp{start_range_dependencies};
            const auto end_range_dependencies_tmp{end_range_dependencies};

            while(start_range_dependencies != end_range_dependencies)
            {
                parent_dependencies.push_back(&(*start_range_dependencies)->ccomponent(0).clocals(0));
                // parent_groups.push_back(start_range_dependencies->get());
                ++start_range_dependencies;
            }

            start_range_dependencies = --end_range_dependencies;
            // clang-format off
#pragma omp task untied default(none) firstprivate(group_child, start_range_dependencies_tmp, end_range_dependencies_tmp, level_interpolator_index)                                         \
  shared(approximation)                                                                      \
    depend(iterator(it = 0 : std::size(group_child->csymbolics().group_dependencies_l2l_in)),                          \
           in : (group_child->csymbolics().group_dependencies_l2l_in.at(it))[0]) depend(inout  : group_child_raw[0])   \
    priority(prio)
            // clang-format on
            {   // Can be a task(in:iterParticles, out:iterChildCells ...)
                std::vector<ptr_parent_groups_type> parent_groups;
                while(start_range_dependencies_tmp != end_range_dependencies_tmp)
                {
                    parent_groups.push_back(start_range_dependencies_tmp->get());
                    ++start_range_dependencies_tmp;
                }
                for(std::size_t cell_index = 0; cell_index < group_child->csymbolics().number_of_component_in_group;
                    ++cell_index)
                {
                    auto& child_cell = group_child->component(cell_index);
                    static constexpr auto number_of_child = math::pow(2, dimension);

                    auto child_morton_index{child_cell.index()};
                    auto parent_morton_index{child_morton_index >> dimension};

                    for(auto p: parent_groups)
                    {
                        if(p->is_inside(parent_morton_index))
                        {
                            int parent_index_in_group = p->component_index(parent_morton_index);
                            assertm(parent_index_in_group != -1, "Upward pass: parent cell not found!");
                            auto const& parent_cell = p->ccomponent(std::size_t(parent_index_in_group));

                            const std::size_t child_index = child_morton_index & (number_of_child - 1);
                            l2l(approximation, parent_cell, child_index, child_cell, level_interpolator_index);
                        }
                    }
                }
            }   // end pragma task
            ++group_of_child_cell_begin;
            }   // end while
        assert(group_of_child_cell_begin == group_of_child_cell_end);
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

        for(std::size_t level = top_height; level < leaf_level; ++level)
        {
            downward_level(level, tree, approximation);
        }
    }
}   // namespace scalfmm::algorithms::omp::pass

#endif   // _OPENMP
#endif   // SCALFMM_ALGORITHMS_OMP_DOWNWARD_HPP
