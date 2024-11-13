// --------------------------------
// See LICENCE file at project root
// File : header.hpp
// --------------------------------
#ifndef SCALFMM_TREE_SYMBOLICS_HPP
#define SCALFMM_TREE_SYMBOLICS_HPP

namespace scalfmm::component
{
    template<typename Component>
    struct symbolics_data
    {
    };

    /// @brief This structure stores the required information
    /// to handle out of group interactions.
    ///
    /// @tparam IndexType : the indexing type
    /// @tparam CoordinateType : the coordinate type of the components
    template<typename Iterator_type, typename IndexType = std::size_t>
    struct out_of_block_interaction
    {
        using iterator_type = Iterator_type;
        /// interaction between A and B
        /// morton index of leaf A inside the group
        IndexType inside_index{};
        /// morton index of leaf B outside of the group
        IndexType outside_index{};
        /// index in the group of leaf A inside the group
        int inside_index_in_block{-1};
        /// index in the group of leaf B inside the group
        int outside_index_in_block{-1};
        /// Iterator of leaf B inside the group
        Iterator_type outside_iterator;

        /**
         * @brief Construct a new out of block interaction object between leaves A and B
         *
         * @param mortonA Morton index of leaf A
         * @param mortonB Morton index of leaf B
         * @param posIngroupA position in the group of leaf A
         */
        out_of_block_interaction(std::size_t& mortonA, std::size_t& mortonB, int& posIngroupA)
          : inside_index(mortonA)
          , outside_index(mortonB)
          , inside_index_in_block(posIngroupA)
          , outside_index_in_block(-1)
        {
        }
        out_of_block_interaction() = default;
        ~out_of_block_interaction() = default;
        /**
         * @brief Display teh out_of_block_interaction struct
         *
         * @param os the ostream
         * @param u the out_of_block_interaction struct
         * @return std::ostream&
         */
        inline friend auto operator<<(std::ostream& os, const out_of_block_interaction<Iterator_type, IndexType>& u) -> std::ostream&
        {
            os << "[ (" << u.inside_index << ", " << u.inside_index_in_block << "), (" << u.outside_index << ", "
               << u.outside_index_in_block << ")]";
            ;
            return os;
        };
    };
}   // namespace scalfmm::component

#endif   // SCALFMM_TREE_LEAF_HPP
