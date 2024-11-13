// --------------------------------
// See LICENCE file at project root
// File : container/block.hpp
// --------------------------------
#pragma once

#include <iterator>
#include <ostream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "scalfmm/container/particle.hpp"
#include "scalfmm/meta/utils.hpp"
#include "scalfmm/tree/header.hpp"
#include "scalfmm/tree/leaf_view.hpp"

#include <cpp_tools/colors/colorized.hpp>

namespace scalfmm::container
{
    template<typename Particle, bool IsConst>
    class proxy_particle_iterator;
    /**
     * @brief the container to store all the data inside the group
     *
     * The container contains
     *   |header| the particles inside the group | the symbolic data of the leaves|
     * The particles are stored in a SOA format
     *
     */
    template<typename Header, typename Particle, typename Component>
    struct particles_block
    {
      private:
        /**
         * @brief Get the true particle raw size object
         *
         * @return constexpr std::size_t
         */
        // constexpr std::size_t get_particle_raw_size() const
        static constexpr std::size_t get_particle_raw_size()
        {
            constexpr auto values_size = sizeof(position_value_type) * dimension_size +
                                         sizeof(inputs_value_type) * inputs_size +
                                         sizeof(outputs_value_type) * outputs_size;
            if constexpr(variables_size == 0)
            {
                return values_size;
            }
            else
            {
                // To improve variables_type and tis sizeof is (the largest element type) * the number of elements
                //  so to save memory we compute exactly the size of all elements without the padding
                // variables_type is a tuple
                variables_type t{};
                size_t var_size{0};
                meta::for_each(t, [&var_size](auto const& t) { var_size += sizeof(t); });
                return var_size + values_size;
            }
        }

      public:
        using header_type = Header;
        using particle_type = Particle;

        using symbolics_type = component::symbolics_data<Component>;
        using self_type = particles_block<header_type, particle_type, Component>;

        using iterator = proxy_particle_iterator<particle_type, false>;
        using const_iterator = proxy_particle_iterator<particle_type, true>;
        using tuple_ptr_type = typename iterator::tuple_ptr_type;

        using position_value_type = typename particle_type::position_value_type;
        static constexpr std::size_t dimension_size = particle_type::dimension_size;
        using position_type = typename particle_type::position_type;

        using inputs_value_type = typename particle_type::inputs_value_type;
        static constexpr std::size_t inputs_size = particle_type::inputs_size;
        using inputs_type = typename particle_type::inputs_type;

        using outputs_value_type = typename particle_type::outputs_value_type;
        static constexpr std::size_t outputs_size = particle_type::outputs_size;
        using outputs_type = typename particle_type::outputs_type;

        static constexpr std::size_t variables_size = particle_type::variables_size;
        using variables_type = typename Particle::variables_type;

        static constexpr std::size_t block_particle_raw = get_particle_raw_size();

      public:
        using raw_type = std::byte;
        using block_type = std::vector<raw_type, XTENSOR_DEFAULT_ALLOCATOR(raw_type)>;
        static constexpr std::size_t number_of_elements = particle_traits<particle_type>::number_of_elements;

        particles_block() = default;
        particles_block(particles_block const&) = default;
        particles_block(particles_block&&) = default;
        ~particles_block() = default;

        particles_block(std::size_t nb_particles, std::size_t nb_leaves)
          : m_block(sizeof(header_type) + (get_particle_raw_size() * nb_particles) +
                      (sizeof(symbolics_type) * nb_leaves),
                    std::byte{0})
          , m_header(reinterpret_cast<header_type* const>(&m_block[0]))
          , m_particles_start(&m_block[0] + sizeof(header_type))
          , m_symbolics_start(reinterpret_cast<symbolics_type* const>(&m_block[0] + sizeof(header_type) +
                                                                      (get_particle_raw_size() * nb_particles)))
          , m_nb_particles(nb_particles)

          , m_nb_leaves(nb_leaves)
          , m_raw_size(sizeof(header_type) + (get_particle_raw_size() * nb_particles) +
                       (sizeof(symbolics_type) * nb_leaves))
          , m_tuple_start(init_tuple_start(m_particles_start, m_nb_particles))
        {
        }
        /**
         * @brief  print the data inside the block
         *
         * @param os the stream
         */
        inline auto print_block_data(std::ostream& os) const noexcept -> void
        {
            os << "starting ptr of the elements of a particle: ";
            io::print_seq(os, m_tuple_start);
            os << std::endl;
            auto iter = reinterpret_cast<typename particle_type::position_value_type*>(m_particles_start);
            int i;
            // print positions
            for(i = 0; i < particle_type::dimension_size; ++i)
            {
                print_one_data(os, iter, "position" + std::to_string(i) + "  ");
            }
            auto iter_i = reinterpret_cast<typename particle_type::inputs_value_type*>(iter);
            // print inputs
            for(i = 0; i < particle_type::inputs_size; ++i)
            {
                print_one_data(os, iter_i, "inputs" + std::to_string(i) + "    ");
            }
            auto iter_o = reinterpret_cast<typename particle_type::outputs_value_type*>(iter_i);
            // print outputs
            for(i = 0; i < particle_type::outputs_size; ++i)
            {
                print_one_data(os, iter_o, "outputs" + std::to_string(i) + "   ");
            }
            // print variables
            if constexpr(particle_type::variables_size != 0)
            {
                i = 0;
                using variables_type = typename particle_type::variables_type;
                variables_type tt{};
                // meta::td<variables_type> t ;
                meta::for_each(tt,
                               [&os, &iter_o, &i, this](auto v)
                               {
                                   auto iter_t = reinterpret_cast<decltype(v)*>(iter_o);
                                   print_one_data(os, iter_t, "variables" + std::to_string(i++) + " ");
                                   iter_o = reinterpret_cast<decltype(iter_o)>(iter_t);
                               });
            }
        }
        /**
         * @brief Display the block information and the data inside the block
         *
         * @param os the stream
         * @param block the current block
         * @return std::ostream&
         */
        inline friend auto operator<<(std::ostream& os, const particles_block& block) -> std::ostream&
        {
            // os << cpp_tools::colors::red;
            os << " sizeof(header_type): " << sizeof(header_type) << std::endl;
            os << " number_of_elements: " << number_of_elements << std::endl;
            os << " sizeof(symbolics_type): " << sizeof(symbolics_type) << std::endl;
            os << " nb_particles: " << block.size() << " nb_leaves:    " << block.nb_leaves() << std::endl;
            os << " raw_size: " << block.raw_size() << std::endl;
            os << " data:     " << block.data() << std::endl;
            os << " particles_start: " << block.cparticles() << " symbolics_start: " << block.csymbolics() << std::endl;
            // os << cpp_tools::colors::reset;
            block.print_block_data(os);
            if(block.data() + sizeof(header_type) != block.particles())
            {
                throw std::runtime_error("Block  : wrong particles_start.");
                std::exit(-1);
            }
            return os;
        }

        inline auto raw_size() const noexcept -> std::size_t { return m_raw_size; }
        inline auto size() const noexcept -> std::size_t { return m_nb_particles; }
        inline auto nb_leaves() const noexcept -> std::size_t { return m_nb_leaves; }
        inline auto data() const noexcept -> const std::byte* { return m_block.data(); }

        inline auto block() const noexcept -> block_type const& { return m_block; }
        inline auto cblock() const noexcept -> block_type const& { return m_block; }
        inline auto block() noexcept -> block_type& { return m_block; }

        inline auto header() const noexcept -> header_type const& { return *m_header; }
        inline auto cheader() const noexcept -> header_type const& { return *m_header; }
        inline auto header() noexcept -> header_type& { return *m_header; }

        inline auto particles() const noexcept -> raw_type* const { return m_particles_start; }
        inline auto cparticles() const noexcept -> const raw_type* const { return m_particles_start; }
        inline auto particles() noexcept -> raw_type* const { return m_particles_start; }

        /**
          * @brief Get the pointer of the first position inside the block
          *
          * This pointer allows to access the different positions as arrays.
          * \code
          *   auto * pos_x = ptr_on_position() ;
          *   auto * pos_y = pos_x + nb_particles;
 -        * \endcode
          * @return particle_type::inputs_value_type* const
          */
        inline auto ptr_on_position() const noexcept -> typename particle_type::position_value_type* const
        {
            return reinterpret_cast<typename particle_type::inputs_value_type*>(m_particles_start);
        }

        inline auto cptr_on_position() const noexcept -> typename particle_type::position_value_type const* const
        {
            return reinterpret_cast<typename particle_type::inputs_value_type*>(m_particles_start);
        }
        /**
         * @brief Get the pointer of the first input inside the block
         *
         * This pointer allows to access the different inputs as arrays.
         * \code
         *   auto * first_input = ptr_on_input() ;
         *   auto *second_input = first_input + nb_particles;
-        * \endcode
         * @return particle_type::inputs_value_type* const
         */
        inline auto ptr_on_input() const noexcept -> typename particle_type::inputs_value_type* const
        {
            return reinterpret_cast<typename particle_type::inputs_value_type*>(
              std::get<dimension_size>(m_tuple_start));
        }
        /**
         * @brief Get the pointer of the first input inside the block
         *
         * This pointer allows to access the different inputs as arrays.
         * \code
         *   auto * first_input = ptr_on_input() ;
         *   auto *second_input = first_input + nb_particles;
-        * \endcode
         *
         * @return particle_type::inputs_value_type* const
         */
        inline auto cptr_on_input() const noexcept -> typename particle_type::inputs_value_type const* const
        {
            return reinterpret_cast<typename particle_type::inputs_value_type*>(
              std::get<dimension_size>(m_tuple_start));
        }
        /**
         * @brief Get the pointer of the first output inside the block
         *
         * This pointer allows to access the different outputs as arrays.
         * \code
         *   auto * first_output = ptr_on_output() ;
         *   auto *second_output = first_output+ nb_particles;
         * \endcode
         * @return outputs_value_type* const
         */
        inline auto ptr_on_output() const noexcept -> outputs_value_type* const
        {
            return reinterpret_cast<outputs_value_type*>(std::get<dimension_size + inputs_size>(m_tuple_start));
        }
        /**
         * @brief Get the pointer of the first output inside the block
         *
         * This pointer allows to access the different outputs as arrays.
         * \code
         *   auto * first_output = ptr_on_output() ;
         *   auto *second_output = first_output+ nb_particles;
         * \endcode
         *
         * @return outputs_value_type* const
         **/
        ///
        inline auto cptr_on_output() const noexcept -> const outputs_value_type* const
        {
            return reinterpret_cast<outputs_value_type*>(std::get<dimension_size + inputs_size>(m_tuple_start));
        }
        /**
         * @brief  Sets all outputs inside the block to zero.
         *
         */
        inline auto reset_outputs() const noexcept -> void
        {
            auto output = reinterpret_cast<outputs_value_type*>(std::get<dimension_size + inputs_size>(m_tuple_start));

            // memset(output, 0, m_nb_particles * outputs_size * sizeof(outputs_value_type));

            for(std::size_t i = 0; i < m_nb_particles * outputs_size; ++i)
            {
                output[i] = outputs_value_type(0.0);
            }
        }

        inline auto symbolics() const noexcept -> const symbolics_type* const { return m_symbolics_start; }
        inline auto csymbolics() const noexcept -> const symbolics_type* const { return m_symbolics_start; }
        inline auto symbolics() noexcept -> symbolics_type* const { return m_symbolics_start; }

        inline auto symbolics(std::size_t i) const noexcept -> symbolics_type const&
        {
            return *(m_symbolics_start + i);
        }
        inline auto csymbolics(std::size_t i) const noexcept -> symbolics_type const&
        {
            return *(m_symbolics_start + i);
        }
        inline auto symbolics(std::size_t i) noexcept -> symbolics_type& { return *(m_symbolics_start + i); }

      public:
        /**
         * @brief to get a tuple of iterator on the beginning all the elements of a particles (position, inputs,
         * outputs)
         *
         * @return iterator
         */
        inline auto begin() noexcept -> iterator { return {m_tuple_start, 0}; }
        /**
         * @brief to get a tuple of iterators on the last particle (position, inputs, outputs)
         *
         * @return iterator
         */
        inline auto end() noexcept -> iterator { return {m_tuple_start, m_nb_particles}; }

        inline auto begin() const noexcept -> const_iterator { return {m_tuple_start, 0}; }
        inline auto cbegin() const noexcept -> const_iterator { return {m_tuple_start, 0}; }

        inline auto end() const noexcept -> const_iterator { return {m_tuple_start, m_nb_particles}; }
        inline auto cend() const noexcept -> const_iterator { return {m_tuple_start, m_nb_particles}; }

        [[nodiscard]] inline auto operator[](std::size_t i)
        {
            auto it = this->begin() + i;
            return *it;
        }
        [[nodiscard]] inline auto operator[](std::size_t i) const
        {
            auto it = this->cbegin() + i;
            return *it;
        }

      private:
        /**
         * @brief
         *
         * @param start_in  position of the first position in the block (byte*)
         * @param nb_particles number of particles
         * @return tuple_ptr_type
         */
        inline constexpr auto init_tuple_start(raw_type* const start_in,
                                               std::size_t nb_particles) const noexcept -> tuple_ptr_type
        {
            auto start{start_in};
            tuple_ptr_type tuple_start{};
            meta::repeat(
              [&start, nb_particles](auto& p)
              {
                  p = reinterpret_cast<decltype(p)>(start);
                  start = reinterpret_cast<raw_type*>(p + nb_particles);
              },
              tuple_start);

            return tuple_start;
        }
        template<std::size_t... Is>
        inline constexpr auto generate_ptr_tuple(std::index_sequence<Is...> /*unused*/) const noexcept
        {
            return tuple_ptr_type((reinterpret_cast<std::tuple_element_t<Is, tuple_ptr_type>>(m_particles_start) +
                                   (Is * m_nb_particles))...);
        }

        template<typename Iterator_type>
        inline auto print_one_data(std::ostream& os, Iterator_type& iter, std::string&& string) const noexcept -> void
        {
            os << string << " (" << m_nb_particles << ") [";
            if(m_nb_particles == 0)
            {
                os << "] ";
            }
            else
            {
                for(int j = 0; j < m_nb_particles - 1; ++j)
                {
                    os << *iter << ", ";
                    ++iter;
                }
                os << *iter << "]\n";
                ++iter;
            }
        }
        template<typename Iterator_type, typename Value_type>
        inline auto init_data(Iterator_type& iter, Value_type&& value) const noexcept -> void
        {
            if(m_nb_particles == 0)
            {
                return;
            }
            for(int j = 0; j < m_nb_particles; ++j)
            {
                *iter = value;
                ++iter;
            }
        }

        //     private:
        /// the block container
        block_type m_block{};
        /// pointer on the beginning  of the header
        header_type* const m_header{nullptr};
        /// pointer on the beginning  of the storage of the particles
        raw_type* const m_particles_start{nullptr};   ///< pointer on the beginning  of the storage of the particles
        symbolics_type* const m_symbolics_start{
          nullptr};                             ///< pointer on the beginning  of the symbolic part of the leaves
        const std::size_t m_nb_particles{0};    ///< Number of particles in the block
        const std::size_t m_nb_leaves{0};       ///< Number of leaves in the block
        const std::size_t m_raw_size{0};        ///< size in octet of the block
        const tuple_ptr_type m_tuple_start{};   ///< tuple of pointers indicating the beginning of the storage of the
                                                ///< elements of the particles
    };

    // the iterator on the container
    /**
     * @brief the iterator on the block container
     *
     * the iterator is a pair (iterators , index). The true iterator is iterators + index
     *  iterators is an iterator tuple pointing to the first particle of the block
     *  index is an int to advance to the position of the good particle in the block
     */
    template<typename Particle, bool IsConst>
    struct proxy_particle_iterator
    {
        using particle_type = Particle;

        using position_value_type = typename particle_type::position_value_type;
        static constexpr std::size_t dimension_size = particle_type::dimension_size;
        using position_type = typename particle_type::position_type;

        using inputs_value_type = typename particle_type::inputs_value_type;
        static constexpr std::size_t inputs_size = particle_type::inputs_size;
        using inputs_type = typename particle_type::inputs_type;

        using outputs_value_type = typename particle_type::outputs_value_type;
        static constexpr std::size_t outputs_size = particle_type::outputs_size;
        using outputs_type = typename particle_type::outputs_type;

        static constexpr std::size_t variables_size = particle_type::variables_size;

        static constexpr std::size_t number_of_elements = dimension_size + inputs_size + outputs_size + variables_size;

        using variables_type = typename particle_type::variables_type;
        using variables_ptr_type = meta::replace_inner_tuple_type_t<std::add_pointer_t, variables_type>;
        using variables_ref_type = meta::replace_inner_tuple_type_t<std::add_lvalue_reference_t, variables_type>;
        using variables_const_ref_type = meta::replace_inner_tuple_type_t<std::add_const_t, variables_ref_type>;
        using tuple_ptr_type = typename meta::cat<
          typename meta::pack_expand_tuple<meta::pack<dimension_size, std::add_pointer_t<position_value_type>>,
                                           meta::pack<inputs_size, std::add_pointer_t<inputs_value_type>>,
                                           meta::pack<outputs_size, std::add_pointer_t<outputs_value_type>>>,
          variables_ptr_type>::type;
        using tuple_ref_type = typename meta::cat<
          typename meta::pack_expand_tuple<meta::pack<dimension_size, std::add_lvalue_reference_t<position_value_type>>,
                                           meta::pack<inputs_size, std::add_lvalue_reference_t<inputs_value_type>>,
                                           meta::pack<outputs_size, std::add_lvalue_reference_t<outputs_value_type>>>,
          variables_ref_type>::type;

        using tuple_const_ref_type = typename meta::cat<
          typename meta::pack_expand_tuple<
            meta::pack<dimension_size, std::add_lvalue_reference_t<std::add_const_t<position_value_type>>>,
            meta::pack<inputs_size, std::add_lvalue_reference_t<std::add_const_t<inputs_value_type>>>,
            meta::pack<outputs_size, std::add_lvalue_reference_t<std::add_const_t<outputs_value_type>>>>,
          variables_const_ref_type>::type;

        using range_position_type = meta::make_range_sequence<0, dimension_size>;
        using range_inputs_type = meta::make_range_sequence<dimension_size, dimension_size + inputs_size>;
        using range_outputs_type =
          meta::make_range_sequence<dimension_size + inputs_size, dimension_size + inputs_size + outputs_size>;
        using range_variables_type =
          meta::make_range_sequence<dimension_size + inputs_size + outputs_size,
                                    dimension_size + inputs_size + outputs_size + variables_size>;
        using range_part_type =
          meta::make_range_sequence<0, dimension_size + inputs_size + outputs_size + variables_size>;

      private:
        tuple_ptr_type vec_{};   ///< a tuple of iterators pointing to the first element of the block
        int index_{0};           ///< the index to move the iterator to the element

      public:
        proxy_particle_iterator() = default;
        proxy_particle_iterator(proxy_particle_iterator const&) = default;
        proxy_particle_iterator(proxy_particle_iterator&&) noexcept = default;
        inline auto operator=(proxy_particle_iterator const&) -> proxy_particle_iterator& = default;
        inline auto operator=(proxy_particle_iterator&&) noexcept -> proxy_particle_iterator& = default;
        ~proxy_particle_iterator() = default;

        proxy_particle_iterator(tuple_ptr_type vec, int index) noexcept
          : vec_{vec}
          , index_{index}
        {
        }

        proxy_particle_iterator(proxy_particle_iterator<Particle, not(IsConst)> const& other) noexcept
          : vec_{other.data()}
          , index_{other.index()}
        {
        }

        inline friend auto operator<<(std::ostream& os, proxy_particle_iterator iter) -> std::ostream&
        {
            auto tup = iter.data();
            auto index = iter.index();

            auto print_tuple = [&os](auto const& tuples)
            {
                os << "[";
                meta::for_each(tuples, [&os](auto const& v) { os << v << ", "; });
                os << "] ";
            };
            auto print_tuple_ptr = [&os](auto const& tuples)
            {
                os << "[";
                meta::for_each(tuples, [&os](auto const& v) { os << &v << ", "; });
                os << "]";
            };
            os << cpp_tools::colors::red;
            os << " proxy: ";
            io::print_ptr_seq(os, *iter);
            os << "=  (";
            io::print_seq(os, tup);
            os << ") index: " << index;
            os << cpp_tools::colors::reset;
            return os;
        }
        // Need to adapt iterator category !!!
        using iterator_category = std::random_access_iterator_tag;
        using value_type = particle_type;
        using reference = std::conditional_t<IsConst, tuple_const_ref_type, tuple_ref_type>;
        using pointer = void;
        using difference_type = std::size_t;
        static constexpr bool is_const_qualified{IsConst};

      private:
        template<size_t... Is>
        [[nodiscard]] inline auto make_proxy(std::index_sequence<Is...> s) const noexcept
        {
            //  return std::forward_as_tuple(*(std::get<Is>(vec_) + index_)...);
            return std::forward_as_tuple(std::get<Is>(vec_)[index_]...);
        }

      public:
        [[nodiscard]] inline auto operator*() const noexcept
        {
            // std::cout << " ---****---- index_ " << index_<<std::endl;
            return make_proxy(std::make_index_sequence<number_of_elements>{});
        }

        [[nodiscard]] inline auto data() const noexcept -> tuple_ptr_type { return this->vec_; }

        [[nodiscard]] inline auto index() const noexcept -> int { return this->index_; }

        [[nodiscard]] inline auto operator==(proxy_particle_iterator const& rhs) const noexcept -> bool

        {
            return index_ == rhs.index_;
        }

        [[nodiscard]] inline auto operator!=(proxy_particle_iterator const& rhs) const noexcept -> bool
        {
            return !(*this == rhs);
        }

        [[nodiscard]] inline auto operator<(proxy_particle_iterator const& rhs) const noexcept -> bool
        {
            return index_ < rhs.index_;
        }

        [[nodiscard]] inline auto operator>(proxy_particle_iterator const& rhs) const noexcept -> bool
        {
            return rhs < *this;
        }
        [[nodiscard]] inline auto operator<=(proxy_particle_iterator const& rhs) const noexcept -> bool
        {
            return !(rhs < *this);
        }

        [[nodiscard]] inline auto operator>=(proxy_particle_iterator const& rhs) const noexcept -> bool
        {
            return !(*this < rhs);
        }

        inline auto operator++() noexcept -> proxy_particle_iterator& { return ++index_, *this; }
        inline auto operator--() noexcept -> proxy_particle_iterator& { return --index_, *this; }
        inline auto operator++(int) noexcept -> proxy_particle_iterator
        {
            const auto old = *this;
            return ++index_, old;
        }
        inline auto operator--(int) noexcept -> proxy_particle_iterator
        {
            const auto old = *this;
            return --index_, old;
        }

        inline auto operator+=(int shift) noexcept -> proxy_particle_iterator& { return index_ += shift, *this; }
        inline auto operator-=(int shift) noexcept -> proxy_particle_iterator& { return index_ -= shift, *this; }

        inline auto operator+(int shift) const noexcept -> proxy_particle_iterator
        {
            return proxy_particle_iterator(vec_, index_ + shift);
        }
        inline auto operator-(int shift) const noexcept -> proxy_particle_iterator
        {
            return proxy_particle_iterator(vec_, index_ - shift);
        }
        inline auto operator-(proxy_particle_iterator const& rhs) const noexcept -> int { return index_ - rhs.index_; }
    };

}   // namespace scalfmm::container
