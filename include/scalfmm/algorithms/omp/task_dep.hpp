﻿// --------------------------------
// See LICENCE file at project root
// File : algorithm/omp/task_dep.hpp
// --------------------------------
#ifndef SCALFMM_ALGORITHMS_OMP_TASK_DEP_HPP
#define SCALFMM_ALGORITHMS_OMP_TASK_DEP_HPP

#ifdef _OPENMP

#include <chrono>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <vector>

#include <cpp_tools/timers/simple_timer.hpp>

#include "scalfmm/algorithms/common.hpp"
#include "scalfmm/algorithms/omp/cell_to_leaf.hpp"
#include "scalfmm/algorithms/omp/direct.hpp"
#include "scalfmm/algorithms/omp/downward.hpp"
#include "scalfmm/algorithms/omp/leaf_to_cell.hpp"
#include "scalfmm/algorithms/omp/periodic.hpp"
#include "scalfmm/algorithms/omp/utils.hpp"
#include "scalfmm/lists/omp.hpp"
//#include "scalfmm/algorithms/omp/priorities.hpp"
#include "scalfmm/algorithms/omp/transfer.hpp"
#include "scalfmm/algorithms/omp/upward.hpp"

namespace scalfmm::algorithms::omp
{
    namespace impl
    {
        /**
         * @brief  Openmp task algorithm to compute the interactions
         *
         * @tparam S        type of the options
         * @tparam TreeS    type of the tree source
         * @tparam TreeT    type of the tree target
         * @tparam NearField  Near-field type
         * @tparam FarField Far-field type
         * @param s          the option (omp, seq, ...)
         * @param tree_source  the tree of particle sources
         * @param tree_target   the tree of particle sources
         * @param fmmoperators  the fmm operator
         * @param op_in          the type of computation (nearfield, farfield, all) see @see operators_to_proceed
         */
        template<typename... S, typename TreeS, typename TreeT, typename NearField, typename FarField>
        inline auto task_dep(options::settings<S...> s, TreeS& tree_source, TreeT& tree_target,
                             operators::fmm_operators<NearField, FarField> const& fmmoperators,
                             unsigned int op_in = operators_to_proceed::all) -> void
        {
            static_assert(options::support(s, options::_s(options::timit)), "task_dep algo unsupported options!");
            using timer = cpp_tools::timers::timer<std::chrono::milliseconds>;
            timer time{};
            bool same_tree{false};
            if constexpr(std::is_same_v<TreeS, TreeT>)
            {
                same_tree = (&tree_source == &tree_target);
            }
            //
            auto const& far_field = fmmoperators.far_field();
            auto const& near_field = fmmoperators.near_field();
            auto const& separation_criterion = fmmoperators.near_field().separation_criterion();
            auto const& mutual = fmmoperators.near_field().mutual();
            // const priorities prios(tree.height());
            //  Buffer for m2l optimization : we calculate the inverse fft on the fly and free memory.
            //  get the shape of the buffer used in the m2l pass (get it from the far field)
            //  and allocate it
            // if(omp_get_max_task_priority() < priorities::max)
            // {
            //     std::cout << cpp_tools::colors::red
            //               << "WARNING the task priorities are not (fully) available. set OMP_MAX_TASK_PRIORITY to "
            //               << priorities::max + 1 << cpp_tools::colors::reset << std::endl;
            // }

            const auto op = tree_target.height() == 2 ? op_in & operators_to_proceed::p2p : op_in;

            if constexpr(options::has(s, options::timit))
            {
                time.tic();
            }
            auto buffers{init_buffers(far_field.approximation())};
            //
#pragma omp parallel default(none)                                                                                     \
  shared(tree_source, tree_target, far_field, near_field, buffers, op, separation_criterion, mutual, same_tree)
            {
#pragma omp single nowait
                {
                    if(tree_target.is_interaction_p2p_lists_built() == false)
                    {
                        list::omp::build_p2p_interaction_list(tree_source, tree_target, separation_criterion, mutual);
                    }
                    if((op & operators_to_proceed::p2p) == operators_to_proceed::p2p)
                    {
                        pass::direct(tree_source, tree_target, near_field);
                    }
                    if(tree_target.is_interaction_m2l_lists_built() == false)
                    {
                        list::omp::build_m2l_interaction_list(tree_source, tree_target, separation_criterion);
                    }
                    if((op & operators_to_proceed::p2m) == operators_to_proceed::p2m)
                    {
                        // std::cout << "\n\n leaf_to_cell pass \n" << std::flush;
                        
                        pass::leaf_to_cell(tree_source, far_field);

                    }

                    if((op & operators_to_proceed::m2m) == operators_to_proceed::m2m)
                    {
                        // std::cout << "\n\n upward pass\n " << std::flush;

                        pass::upward(tree_source, far_field.approximation());
                    }

                    if(same_tree && tree_target.box().is_periodic())
                    {
                        //  timers["field0"].tic();
                        pass::build_field_level0(tree_target, far_field.approximation());
                        //  timers["field0"].tac();
                    }
                    if((op & operators_to_proceed::m2l) == operators_to_proceed::m2l)
                    {
                        // std::cout << "\n\n transfer(pass::split_m2l::remove_leaf_level)  pass\n " << std::flush;

                        pass::transfer(tree_source, tree_target, far_field, buffers,
                                       pass::split_m2l::remove_leaf_level);

                    }
                    if((op & operators_to_proceed::l2l) == operators_to_proceed::l2l)
                    {
                        // std::cout << "\n\n downward pass\n " << std::endl << std::flush;

                        pass::downward(tree_target, far_field.approximation());
                    }
                    if((op & operators_to_proceed::m2l) == operators_to_proceed::m2l)
                    {
                        // std::cout << "\n\n transfer(pass::split_m2l::leaf_level)  pass\n " << std::flush;

                        pass::transfer(tree_source, tree_target, far_field, buffers, pass::split_m2l::leaf_level);
                    }
                    if((op & operators_to_proceed::l2p) == operators_to_proceed::l2p)
                    {
                        // std::cout << " cell_to_leaf pass\n " << std::flush;

                        pass::cell_to_leaf(tree_target, far_field);
                    }
                }   // end single
            }   // end parallel

            delete_buffers(buffers);

            if constexpr(options::has(s, options::timit))
            {
                time.tac();
                using duration_type = timer::duration;
                using value_type = double;
                static constexpr value_type unit_multiplier = static_cast<value_type>(duration_type::period::den);
                std::cout << "Full algo : " << std::fixed << std::setprecision(2) << time.cumulated() / unit_multiplier
                          << " s \n";
            }
        }

        template<typename Tree, typename NearField, typename FarField>
        inline auto task_dep(Tree& tree, operators::fmm_operators<NearField, FarField> const& fmmoperators,
                             unsigned int op = operators_to_proceed::all) -> void
        {
            return task_dep(options::settings<>{}, tree, tree, fmmoperators, op);
        }
        template<typename... S, typename Tree, typename NearField, typename FarField>
        inline auto task_dep(options::settings<S...> s, Tree& tree,
                             operators::fmm_operators<NearField, FarField> const& fmmoperators,
                             unsigned int op = operators_to_proceed::all) -> void
        {
            return task_dep(s, tree, tree, fmmoperators, op);
        }
        template<typename TreeS, typename TreeT, typename NearField, typename FarField>
        inline auto task_dep(TreeS& tree_source, TreeT& tree_target,
                             operators::fmm_operators<NearField, FarField> const& fmmoperators,
                             unsigned int op = operators_to_proceed::all) -> void
        {
            return task_dep(options::settings<>{}, tree_source, tree_target, fmmoperators, op);
        }

    }   // namespace impl

    DECLARE_OPTIONED_CALLEE(task_dep);

}   // namespace scalfmm::algorithms::omp

#endif   // _OPENMP
#endif   // SCALFMM_ALGORITHMS_OMP_TASK_DEP_HPP
