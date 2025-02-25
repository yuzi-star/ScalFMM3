// --------------------------------
// See LICENCE file at project root
// File : leaf_view.hpp
// --------------------------------
#ifndef SCALFMM_TREE_LEAF_VIEW_HPP
#define SCALFMM_TREE_LEAF_VIEW_HPP

#include <algorithm>
#include <array>
#include <iterator>
#include <tuple>

#include <cpp_tools/colors/colorized.hpp>

#include "scalfmm/container/access.hpp"
#include "scalfmm/container/particle_container.hpp"
#include "scalfmm/container/point.hpp"
#include "scalfmm/meta/traits.hpp"
#include "scalfmm/meta/type_pack.hpp"
#include "scalfmm/meta/utils.hpp"
#include "scalfmm/tags/tags.hpp"
#include "scalfmm/tree/header.hpp"
#include "scalfmm/utils/io_helpers.hpp"
#include "scalfmm/utils/massert.hpp"
#include "scalfmm/utils/math.hpp"

namespace scalfmm::component
{
    template<typename Component, typename Particle>
    struct group_of_particles;
    // template<typename Particle, typename D=void>
    // class leaf_view;

    /// @brief This is the leaf type stored at the bottom of the tree.
    /// Its stores the particle container and some symbolics informations.
    ///
    /// @tparam Particle : the type of particle to store.
    template<typename Particle, typename D = void>
    class leaf_view
    {
      public:
        // static dimension of the particle
        static constexpr std::size_t dimension = Particle::dimension;
        // self type
        using self_type = leaf_view<Particle>;
        // the particle type stored in the container
        using particle_type = Particle;
        using proxy_type = typename Particle::proxy_type;
        using const_proxy_type = typename Particle::const_proxy_type;
        // Symbolic types : the type storing informations about the leaf component
        using symbolics_type = symbolics_data<std::conditional_t<std::is_void_v<D>,self_type,D> >;
        // the position type extract from the symbolic information
        using position_type = typename container::particle_traits<particle_type>::position_type;
        // the position value type
        using value_type = typename position_type::value_type;
        using position_coord_type = typename position_type::value_type;
        using inputs_type = typename container::particle_traits<particle_type>::inputs_type::value_type;
        using outputs_type = typename container::particle_traits<particle_type>::outputs_type::value_type;
        //
        using group_type = typename symbolics_type::group_type;
        // the particle container type
        using storage_type = typename group_type::storage_type;
        using iterator = typename storage_type::iterator;
        using const_iterator = typename storage_type::const_iterator;
        // the number of inputs
        static constexpr std::size_t inputs_size = container::particle_traits<particle_type>::inputs_size;

        // rule of five generated by the compiler
        leaf_view() = default;
        leaf_view(leaf_view const&) = default;
        leaf_view(leaf_view&&) noexcept = default;
        inline auto operator=(leaf_view const&) -> leaf_view& = default;
        inline auto operator=(leaf_view&&) noexcept -> leaf_view& = default;
        ~leaf_view() = default;

        /// @brief Constuctors from center and width
        ///
        /// @param center : the center of the leaf
        /// @param width : the width of the feaf
        ///

        /**
         * @brief Construct a new leaf view object
         *
         * @param particles_range is a pair of iterators (begin, end) in the block to access to the particles in the leaf
         * @param symbolics_ptr   is a pointer on the symbolic structure of the leaf
         */
        leaf_view(std::pair<iterator, iterator> particles_range, symbolics_type* const symbolics_ptr)
          : m_particles_range(particles_range)
          , m_const_particles_range(particles_range)
          , m_symbolics(symbolics_ptr)
        {
        }
        leaf_view(std::pair<const_iterator, const_iterator> particles_range, symbolics_type* const symbolics_ptr)
          : m_particles_range(particles_range)
          , m_const_particles_range(particles_range)
          , m_symbolics(symbolics_ptr)
        {
        }
        /// @brief Returns the center of the leaf
        ///
        /// @return position_type
        [[nodiscard]] inline auto center() const noexcept -> position_type const& { return m_symbolics->center; }
        [[nodiscard]] inline auto center() noexcept -> position_type& { return m_symbolics->center; }
        /// @brief Returns the width of the leaf
        ///
        /// @return value_type
        [[nodiscard]] inline auto width() const noexcept -> value_type { return m_symbolics->width; }
        [[nodiscard]] inline auto width() noexcept -> value_type& { return m_symbolics->width; }

        [[nodiscard]] inline auto begin() -> iterator { return m_particles_range.first; }
        [[nodiscard]] inline auto begin() const -> const_iterator { return const_iterator(m_particles_range.first); }
        [[nodiscard]] inline auto cbegin() const -> const_iterator { return const_iterator(m_particles_range.first); }

        [[nodiscard]] inline auto end() -> iterator { return m_particles_range.second; }
        [[nodiscard]] inline auto end() const -> const_iterator { return const_iterator(m_particles_range.second); }
        [[nodiscard]] inline auto cend() const -> const_iterator { return const_iterator(m_particles_range.second); }

        /// @brief Return the number of particles
        ///
        /// @return std::size_t
        [[nodiscard]] inline auto size() const noexcept -> std::size_t
        {
            return std::distance(m_particles_range.first, m_particles_range.second);
        }

        /// @brief Non const accessor on the container
        ///
        /// @return a reference on the particle_container_type
        [[nodiscard]] inline auto particles() -> std::pair<iterator, iterator> const& { return m_particles_range; }
        /// @brief Const accessor on the container
        ///
        /// @return a const reference on the particle_container_type
        [[nodiscard]] inline auto particles() const noexcept -> std::pair<const_iterator, const_iterator> const&
        {
            return m_const_particles_range;
        }
        // { return std::make_pair(const_iterator(m_particles_range.first)
        //             , const_iterator(m_particles_range.second)); }
        /// @brief Const accessor on the container
        ///
        /// @return a const reference on the particle_container_type
        [[nodiscard]] inline auto cparticles() const noexcept -> std::pair<const_iterator, const_iterator> const&
        // {
        //     return std::make_pair(const_iterator(m_particles_range.first)
        //             , const_iterator(m_particles_range.second));
        // }
        {
            return m_const_particles_range;
        }
        /// @brief Indexed accessor on the particle container
        ///
        /// @param i : the index at which the access is performed.
        ///
        /// @return a particle_type reference
        [[nodiscard]] inline auto particle(std::size_t i) noexcept -> proxy_type { return proxy_type(*(m_particles_range.first + int(i))); }
        /// @brief Indexed accessor on the particle container
        ///
        /// @param i : the index at which the access is performed.
        ///
        /// @return a particle_type const reference
        [[nodiscard]] inline auto particle(std::size_t i) const noexcept -> const_proxy_type { return const_proxy_type(*(m_particles_range.first + int(i))); }

        /// @brief subscript operator
        ///
        /// @param i : the index at which the access is performed.
        ///
        /// @return a particle_type reference
        [[nodiscard]] inline auto operator[](std::size_t i) noexcept -> proxy_type
        {
            return proxy_type(*(m_particles_range.first + int(i)));
        }

        /// @brief subscript operator
        ///
        /// @param i : the index at which the access is performed.
        ///
        /// @return a particle_type const reference
        [[nodiscard]] inline auto operator[](std::size_t i) const noexcept -> const_proxy_type
        {
            return const_proxy_type(*(m_particles_range.first + int(i)));
        }

        /// @brief Access to the symbolic type
        ///
        /// @return a symbolics_type reference
        [[nodiscard]] inline auto symbolics() noexcept -> symbolics_type& { return *m_symbolics; }
        /// @brief Access to the symbolic type
        ///
        /// @return a const symbolics_type reference
        [[nodiscard]] inline auto symbolics() const noexcept -> symbolics_type const& { return *m_symbolics; }
        /// @brief Access to the symbolic type
        ///
        /// @return a const symbolics_type reference
        [[nodiscard]] inline auto csymbolics() const noexcept -> symbolics_type const& { return *m_symbolics; }

        /// @brief Returns the morton index of the leaf
        ///
        /// @return std::size_t
        [[nodiscard]] inline auto index() const noexcept -> std::size_t { return m_symbolics->morton_index; }
        [[nodiscard]] inline auto index() noexcept -> std::size_t& { return m_symbolics->morton_index; }

        inline friend auto operator<<(std::ostream& os, const leaf_view& leaf) -> std::ostream&
        {
            os << cpp_tools::colors::blue;
            os << "  index: " << leaf.index() << " nb_part: " << leaf.size() << std::endl;
            os << " begin: ";
            io::print_ptr_seq(os, *leaf.particles().first);
            os << "\n   end: ";
            io::print_ptr_seq(os, *leaf.particles().second);
            os << "\n  dist: " << std::distance(leaf.particles().first, leaf.particles().second);
            os << cpp_tools::colors::reset;
            return os;
        }

        ///
        /// \brief reset_outputs reset  outputs in the leaf
        ///
        // inline void reset_outputs()
        // {
        //     //
        //     m_particles.reset_outputs();
        // }
        inline void reset_outputs()
        {
            std::clog << "leaf_view::reset_outputs not yet implemented.\n ";
            // // leaf level update only P2P ???
            // component::for_each(std::get<0>(begin()), std::get<0>(end()),
            //                     [this](auto& group)
            //                     {
            //                         std::size_t index_in_group{0};
            //                         component::for_each(std::begin(*group), std::end(*group),
            //                                             [&group, &index_in_group, this](auto& leaf)
            //                                             { leaf.reset_outputs(); });
            //                     });
        }

      private:
        /// The particle container
        ///  (begin, end) the iterators to access the particles in the group container
        /// iterator est une proxy_particle_iterator and the end operator has a lazy
        /// evaluation. This means that you HAVE to use *end to evaluate it
        std::pair<iterator, iterator> m_particles_range{};
        std::pair<const_iterator, const_iterator> m_const_particles_range{};
        /// a pointer to he symbolic type
        symbolics_type* m_symbolics{nullptr};
    };

    /// @brief The symbolics type stores information about the leaf
    /// It represents a generic that also exists on the cells
    ///
    /// @tparam P
    template<typename P>
    struct symbolics_data<leaf_view<P>>
    {
        // the group type
        using group_type = group_of_particles<leaf_view<P>, P>;
        // the leaf type
        using component_type = leaf_view<P>;
        // the position value type
        using position_value_type = typename container::particle_traits<P>::position_value_type;
        // the position type
        using position_type = container::point<position_value_type, component_type::dimension>;
        // the coordinate type to store the coordinate in the tree
        // using coordinate_type =
        //  decltype(index::get_coordinate_from_morton_index<component_type::dimension>(std::size_t{}));
        // the number of interactions of the leaf
        static constexpr std::size_t number_of_interactions{math::pow(3, component_type::dimension)};
        // type of the array storing the indexes of the theoretical interaction list
        using interaction_index_array_type = std::array<std::size_t, number_of_interactions>;
        // type of the array storing the iterators of the interacting leaves available in the current group
        // using iterator_type = typename group_type::iterator_type;
        using seq_iterator_type =
          std::conditional_t<meta::exist_v<meta::inject<group_type>>, meta::exist_t<meta::inject<group_type>>,
                             std::tuple<typename group_type::iterator_type, group_type>>;
        using iterator_source_type = std::tuple_element_t<0, seq_iterator_type>;
        using iterator_array_type = std::array<iterator_source_type, number_of_interactions>;

        // the array storing the indexes of the theoretical interaction list
        interaction_index_array_type interaction_indexes{};
        // the array storing the iterators of the interacting leaves available in the current group
        iterator_array_type interaction_iterators{};
        // the morton index of the leaf
        std::size_t morton_index{0};
        // the theoretical number of neighbors.
        std::size_t number_of_neighbors{0};
        // the number of numbers available in the group of the leaf
        std::size_t existing_neighbors_in_group{0};
        // Position of the center
        position_type center{};
        // The width of the leaf
        position_value_type width{};

        void set(int counter, std::size_t const& idx, const iterator_source_type& leaf_iter)
        {
            interaction_iterators.at(counter) = leaf_iter;
        }
        void finalize(bool done, std::size_t const& counter_existing_component)
        {
            number_of_neighbors = counter_existing_component;
        }
    };

}   // namespace scalfmm::component

#endif   // SCALFMM_TREE_LEAF_VIEW_HPP
