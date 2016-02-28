//===----------------------------------------------------------------------===//
//
//                         PelotonDB
//
// BWTree.h
//
// Identification: src/backend/index/BWTree.h
//
// Copyright (c) 2015, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#pragma once
#include <limits>
#include <vector>
#include <atomic>
#include <algorithm>
#include <cassert>
#include "backend/storage/tuple.h"
#include "backend/common/logger.h"

#define BWTREE_DEBUG

#ifdef BWTREE_DEBUG

#define bwt_printf(fmt, ...)                              \
  do {                                                    \
    printf("%-24s(): " fmt, __FUNCTION__, ##__VA_ARGS__); \
  } while (0);
#define bwt_printf_red(fmt, ...)                                              \
  do {                                                                        \
    printf("\033[1;31m%-24s(): " fmt "\033[0m", __FUNCTION__, ##__VA_ARGS__); \
  } while (0);
#define bwt_printf_redb(fmt, ...)                                             \
  do {                                                                        \
    printf("\033[1;41m%-24s(): " fmt "\033[0m", __FUNCTION__, ##__VA_ARGS__); \
  } while (0);
#define bwt_printf_green(fmt, ...)                                            \
  do {                                                                        \
    printf("\033[1;32m%-24s(): " fmt "\033[0m", __FUNCTION__, ##__VA_ARGS__); \
  } while (0);
#define bwt_printf_greenb(fmt, ...)                                           \
  do {                                                                        \
    printf("\033[1;42m%-24s(): " fmt "\033[0m", __FUNCTION__, ##__VA_ARGS__); \
  } while (0);
#define bwt_printf_blue(fmt, ...)                                             \
  do {                                                                        \
    printf("\033[1;34m%-24s(): " fmt "\033[0m", __FUNCTION__, ##__VA_ARGS__); \
  } while (0);
#define bwt_printf_blueb(fmt, ...)                                            \
  do {                                                                        \
    printf("\033[1;44m%-24s(): " fmt "\033[0m", __FUNCTION__, ##__VA_ARGS__); \
  } while (0);
#else

#define bwt_printf(args...) \
  do {                      \
  } while (0);
#define bwt_printf_red(fmt, ...) \
  do {                           \
  } while (0);
#define bwt_printf_redb(fmt, ...) \
  do {                            \
  } while (0);
#define bwt_printf_green(fmt, ...) \
  do {                             \
  } while (0);
#define bwt_printf_greenb(fmt, ...) \
  do {                              \
  } while (0);
#define bwt_printf_blue(fmt, ...) \
  do {                            \
  } while (0);
#define bwt_printf_blueb(fmt, ...) \
  do {                             \
  } while (0);

#endif

namespace peloton {
namespace index {

// Look up the stx btree interface for background.
// peloton/third_party/stx/btree.h
template <typename KeyType, typename ValueType, typename KeyComparator>
class BWTree {
 private:
  using PID = uint64_t;

  constexpr static PID NONE_PID = std::numeric_limits<PID>::max();
  constexpr static unsigned int max_table_size = 1 << 24;
  // Threshold of delta chain length on an inner node to trigger a consolidate
  constexpr static unsigned int delta_chain_inner_thesh = 8;
  // Threshold of delta chain length on a leaf node to trigger a consolidate
  constexpr static unsigned int delta_chain_leaf_thesh = 8;
  // Node sizes for triggering splits and merges on inner nodes
  constexpr static unsigned int inner_node_size_min = 8;
  constexpr static unsigned int inner_node_size_max = 16;
  // Node sizes for triggering splits and merges on leaf nodes
  constexpr static unsigned int leaf_node_size_min = 8;
  constexpr static unsigned int leaf_node_size_max = 16;
  // Debug constant: The maximum number of iterations we could do
  // It prevents dead loop hopefully
  constexpr static int iter_max = 99999;

  // Enumeration of the types of nodes required in updating both the values
  // and the index in the Bw Tree. Currently only adding node types for
  // supporting splits.
  // TODO: more node types to be added for merging

  enum PageType {
    leaf,
    inner,
    // Page type
    deltaInsert,
    deltaDelete,
    // Inner type & page type
    deltaSplit,
    deltaIndexTermInsert,
    deltaIndexTermDelete,
    deltaRemove,
    deltaMerge,
  };

  enum InstallDeltaResult {
    install_success,
    install_try_again,
    install_node_invalid,
    install_need_consolidate,
  };

  /*
   * class BwNode - Generic node class; inherited by leaf, inner
   * and delta node
   */
  class BwNode {
   public:
    BwNode(PageType _type) : type(_type) {}
    PageType type;
  };  // class BwNode

  /*
   * class BwDeltaNode - Delta page
   */
  class BwDeltaNode : public BwNode {
   public:
    BwDeltaNode(PageType _type, BwNode* _child_node) : BwNode(_type) {
      child_node = _child_node;
    }

    BwNode* child_node;
  };  // class BwDeltaNode

  /*
   * class BwDeltaInsertNode - Key insert delta
   */
  class BwDeltaInsertNode : public BwDeltaNode {
   public:
    BwDeltaInsertNode(BwNode* _child_node,
                      std::pair<KeyType, ValueType> _ins_record)
        : BwDeltaNode(PageType::deltaInsert, _child_node) {
      ins_record = _ins_record;
    }

    std::pair<KeyType, ValueType> ins_record;
  };  // class BwDeltaInsertNode

  /*
   * class BwDeltaDeleteNode - Key delete delta
   */
  class BwDeltaDeleteNode : public BwDeltaNode {
   public:
    BwDeltaDeleteNode(BwNode* _child_node,
                      std::pair<KeyType, ValueType> _del_record)
        : BwDeltaNode(PageType::deltaDelete, _child_node) {
      del_record = _del_record;
    }

    std::pair<KeyType, ValueType> del_record;
  };  // class BwDeltaDeleteNode

  /*
   * class BwDeltaSplitNode - Leaf and inner split node
   */
  class BwDeltaSplitNode : public BwDeltaNode {
   public:
    BwDeltaSplitNode(BwNode* _child_node, KeyType _separator_key,
                     PID _split_sibling, KeyType _next_separator_key)
        : BwDeltaNode(PageType::deltaSplit, _child_node),
          separator_key(_separator_key),
          next_separator_key(_next_separator_key),
          split_sibling(_split_sibling) {}

    KeyType separator_key;
    PID split_sibling;
    KeyType next_separator_key;
  };  // class BwDeltaSplitNode

  /*
   * class BwDeltaIndexTermInsertNode - Index separator add
   */
  class BwDeltaIndexTermInsertNode : public BwDeltaNode {
   public:
    BwDeltaIndexTermInsertNode(BwNode* _child_node,
                               KeyType new_split_separator_key,
                               PID new_split_sibling,
                               KeyType next_separator_key)
        : BwDeltaNode(PageType::deltaIndexTermInsert, _child_node),
          new_split_separator_key(new_split_separator_key),
          new_split_sibling(new_split_sibling),
          next_separator_key(next_separator_key) {}

    KeyType new_split_separator_key;
    PID new_split_sibling;
    KeyType next_separator_key;
  };  // class BwDeltaIndexTermInsertNode

  /*
   * BwDeltaIndexTermDeleteNode - Remove separator in inner page
   */
  class BwDeltaIndexTermDeleteNode : public BwDeltaNode {
   public:
    BwDeltaIndexTermDeleteNode(BwNode* _child_node, PID node_to_merge_into,
                               PID node_to_remove, KeyType merge_node_low_key,
                               KeyType next_separator_key)
        : BwDeltaNode(PageType::deltaIndexTermDelete, _child_node),
          node_to_merge_into(node_to_merge_into),
          node_to_remove(node_to_remove),
          merge_node_low_key(merge_node_low_key),
          next_separator_key(next_separator_key) {}

    PID node_to_merge_into;
    PID node_to_remove;
    KeyType merge_node_low_key;
    KeyType remove_node_low_key;
    KeyType next_separator_key;
  };  // class BwDeltaIndexTermDeleteNode

  /*
   * class BwDeltaRemoveNode - Delete and free page
   *
   * NOTE: This is not delete key node
   */
  class BwDeltaRemoveNode : public BwDeltaNode {
   public:
    BwDeltaRemoveNode(BwNode* _child_node)
        : BwDeltaNode(PageType::deltaRemove, _child_node) {}
  };  // class BwDeltaRemoveNode

  /*
   * class BwDeltaMergeNode - Merge two pages into one
   */
  class BwDeltaMergeNode : public BwDeltaNode {
   public:
    BwDeltaMergeNode(BwNode* _child_node, KeyType separator_key,
                     BwNode* merge_node)
        : BwDeltaNode(PageType::deltaMerge, _child_node),
          separator_key(separator_key),
          merge_node(merge_node) {}

    PID node_to_remove;
    KeyType separator_key;
    BwNode* merge_node;
  };  // class BwDeltaMergeNode

  /*
   * class BwInnerNode - Inner node that contains separator
   *
   * Contains guide post keys for pointing to the right PID when search
   * for a key in the index
   *
   * Elastic container to allow for separation of consolidation, splitting
   * and merging
   */
  class BwInnerNode : public BwNode {
   public:
    BwInnerNode(KeyType _lower_bound)
        : BwNode(PageType::inner), lower_bound(_lower_bound) {}

    const KeyType lower_bound;
    std::vector<std::pair<KeyType, PID>> separators;
  };  // class BwInnerNode

  /*
   * class BwLeafNode - Leaf node that actually stores data
   *
   * Lowest level nodes in the tree which contain the payload/value
   * corresponding to the keys
   */
  class BwLeafNode : public BwNode {
   public:
    BwLeafNode(KeyType _lower_bound, PID _next)
        : BwNode(PageType::leaf), lower_bound(_lower_bound), next(_next) {}

    const KeyType lower_bound;
    PID next;

    // Elastic container to allow for separation of consolidation, splitting
    // and merging
    std::vector<std::pair<KeyType, ValueType>> data;
  };  // class BwLeafNode

  /// //////////////////////////////////////////////////////////////
  /// ValueType Comparator (template type oblivious)
  /// //////////////////////////////////////////////////////////////

  /*
   * class ValueComparator - Compare struct ItemPointer which is always
   *                         used as ValueType
   *
   * If not then compilation error
   */
  class ValueComparator {
    // ItemPointerData -- BlockIdData -- uint16 lo
    //                 |             |-- uint16 hi
    //                 |
    //                 |- OffsetNumber - typedefed as uint16
   public:
    /*
     * operator() - Checks for equality
     *
     * Return True if two data pointers are the same. False otherwise
     * Since we do not enforce any order for data it is sufficient for
     * us just to compare equality
     */
    bool operator()(const ItemPointer &a, const ItemPointer &b) const {
      return (a.block == b.block) && \
             (a.offset == b.offset);
    }
  };

  /// //////////////////////////////////////////////////////////////
  /// Method decarations & definitions
  /// //////////////////////////////////////////////////////////////
 public:
  // TODO: pass a settings structure as we go along instead of
  // passing in individual parameter values
  BWTree(KeyComparator _m_key_less);
  ~BWTree();

  bool insert(const KeyType& key, const ValueType& value);
  bool exists(const KeyType& key);
  bool erase(const KeyType& key, const ValueType& value);

  std::vector<ValueType> find(const KeyType& key);

 private:
  /// True if a < b ? "constructed" from m_key_less()
  inline bool key_less(const KeyType& a, const KeyType& b) const {
    return m_key_less(a, b);
  }

  /// True if a <= b ? constructed from key_less()
  inline bool key_lessequal(const KeyType& a, const KeyType& b) const {
    return !m_key_less(b, a);
  }

  /// True if a > b ? constructed from key_less()
  inline bool key_greater(const KeyType& a, const KeyType& b) const {
    return m_key_less(b, a);
  }

  /// True if a >= b ? constructed from key_less()
  inline bool key_greaterequal(const KeyType& a, const KeyType& b) const {
    return !m_key_less(a, b);
  }

  /// True if a == b ? constructed from key_less(). This requires the <
  /// relation to be a total order, otherwise the B+ tree cannot be sorted.
  inline bool key_equal(const KeyType& a, const KeyType& b) const {
    return !m_key_less(a, b) && !m_key_less(b, a);
  }

  // Internal functions to be implemented
  void traverseAndConsolidateLeaf(
      BwNode* node, std::vector<BwNode*>& garbage_nodes,
      std::vector<std::pair<KeyType, ValueType>>& data, PID& sibling,
      bool& has_merge, BwNode*& merge_node, KeyType& lower_bound);

  bool consolidateLeafNode(PID id);

  void traverseAndConsolidateInner(
      BwNode* node, std::vector<BwNode*>& garbage_nodes,
      std::vector<std::pair<KeyType, PID>>& separators, bool& has_merge,
      BwNode*& merge_node, KeyType& lower_bound);

  bool consolidateInnerNode(PID id);

  bool isSMO(BwNode* n);

  bool isBasePage(BwNode* node_p) { return node_p->type == leaf; }

  PID findLeafPage(const KeyType& key);

  bool splitInnerNode(PID id);

  bool splitLeafNode(PID id);

  bool mergeInnerNode(PID id);

  bool mergeLeafNode(PID id);

  // Atomically install a page into mapping table
  // NOTE: There are times that new pages are not installed into
  // mapping table but instead they directly replace other page
  // with the same PID
  PID installPage(BwNode* new_node_p);

  // This only applies to leaf node - For intermediate nodes
  // the insertion of sep/child pair must be done using different
  // insertion method
  InstallDeltaResult installDeltaInsert(PID leaf_pid, const KeyType& key,
                                        const ValueType& value);

  InstallDeltaResult installDeltaDelete(PID leaf_pid, const KeyType& key,
                                        const ValueType& value);

  InstallDeltaResult installIndexTermDeltaInsert(
      PID pid, const BwDeltaSplitNode* split_node);

  InstallDeltaResult installIndexTermDeltaDelete(PID pid, const KeyType& key,
                                                 const ValueType& value);

  // Functions to install SMO deltas
  InstallDeltaResult installDeltaRemove(PID node);

  InstallDeltaResult installDeltaMerge(PID node, PID sibling);

  InstallDeltaResult installDeltaSplit(PID node);

  /////////////////////////////////////////////////////////////////
  // Data member definition
  /////////////////////////////////////////////////////////////////

  // TODO: Add a global garbage vector per epoch using a lock

  // Note that this cannot be resized nor moved. So it is effectively
  // like declaring a static array
  // TODO: Maybe replace with a static array
  // NOTE: This is updated together with next_pid atomically
  std::atomic<size_t> current_mapping_table_size;

  // Next available PID to allocate for any node
  // This variable is made atomic to facilitate our atomic mapping table
  // implementation
  std::atomic<PID> next_pid;
  std::vector<std::atomic<BwNode*>> mapping_table{max_table_size};

  PID m_root;
  const KeyComparator& m_key_less;

  // Leftmost leaf page
  // NOTE: We assume the leftmost lead page will always be there
  // For split it remains to be the leftmost page
  // For merge and remove we need to make sure the last remianing page
  // shall not be removed
  // Using this pointer we could do sequential search more efficiently
  PID first_leaf;
};

}  // End index namespace
}  // End peloton namespace

// Template definitions
#include <set>
#include <cassert>

namespace peloton {
namespace index {

/*
 * Constructor - Construct a new tree with an single element
 *               intermediate node and empty leaf node
 *
 * NOTE: WE have a corner case here: initially the leaf node
 * is empty, so any leaf page traversal needs to be able to handle
 * empty lead node (i.e. calling data.back() causes undefined behaviour)
 */
template <typename KeyType, typename ValueType, typename KeyComparator>
BWTree<KeyType, ValueType, KeyComparator>::BWTree(KeyComparator _m_key_less)
    : current_mapping_table_size(0), next_pid(0), m_key_less(_m_key_less) {
  // Initialize an empty tree
  BwLeafNode* initial_leaf = new BwLeafNode(KeyType(), NONE_PID);

  PID leaf_pid = installPage(initial_leaf);
  BwInnerNode* initial_inner = new BwInnerNode(KeyType());

  PID inner_pid = installPage(initial_inner);
  initial_inner->separators.push_back({KeyType(), leaf_pid});

  m_root = inner_pid;
  first_leaf = leaf_pid;

  bwt_printf("Init: Initializer returns. Leaf = %lu, inner = %lu\n", leaf_pid,
             inner_pid);

  return;
}

/*
 * Destructor - Free up all pages
 */
template <typename KeyType, typename ValueType, typename KeyComparator>
BWTree<KeyType, ValueType, KeyComparator>::~BWTree() {
  // TODO: finish this
  return;
}

/*
 * isSMO() - Returns true if the target is a SMO operation on leaf page
 *
 * We maintain the invariant that if SMO is going to appear in delta chian,
 * then it must be the first on it. Every routine that sees an SMO on leaf delta
 * must then consolidate it in order to append new delta record
 *
 * SMOs that could appear in leaf delta chain: (ziqi: Please correct me if I'm
 *wrong)
 *   - deltaSplit
 *   - deltaRemove
 *   - deltaMerge
 */
template <typename KeyType, typename ValueType, typename KeyComparator>
bool BWTree<KeyType, ValueType, KeyComparator>::isSMO(BwNode* n) {
  PageType type = n->type;
  return (type == PageType::deltaSplit) || (type == PageType::deltaRemove) ||
         (type == PageType::deltaMerge);
}

/*
 * exists() - Return true if a key exists in the tree
 *
 * Searches through the chain of delta pages, scanning for both
 * delta record and the final leaf record
 *
 * NOTE: Currently this method does not support duplicated key
 * test, since duplicated keys might span multiple pages
 */
template <typename KeyType, typename ValueType, typename KeyComparator>
bool BWTree<KeyType, ValueType, KeyComparator>::exists(const KeyType& key) {
  bwt_printf("key = %d\n", key);
  // Find the first page where the key lies in
  PID page_pid = findLeafPage(key);

  BwNode* leaf_node_p = mapping_table[page_pid].load();
  assert(leaf_node_p != nullptr);

  BwDeltaInsertNode* insert_page_p = nullptr;
  BwDeltaDeleteNode* delete_page_p = nullptr;
  BwLeafNode* base_page_p = nullptr;
  std::pair<KeyType, ValueType>* pair_p = nullptr;

  while (1) {
    if (leaf_node_p->type == deltaInsert) {
      insert_page_p = static_cast<BwDeltaInsertNode*>(leaf_node_p);

      bwt_printf("See DeltaInsert Page: %d\n", insert_page_p->ins_record.first);

      // If we see an insert node first, then this implies that the
      // key does exist in the future consolidated version of the page
      if (key_equal(insert_page_p->ins_record.first, key) == true) {
        return true;
      } else {
        leaf_node_p = (static_cast<BwDeltaNode*>(leaf_node_p))->child_node;
      }
    } else if (leaf_node_p->type == deltaDelete) {
      delete_page_p = static_cast<BwDeltaDeleteNode*>(leaf_node_p);

      // For delete record it implies the node has been removed
      // Ravi: This is wrong in the presence of duplicates you cannot bail out
      // on seeing one delete in the key
      if (key_equal(delete_page_p->del_record.first, key) == true) {
        return false;
      } else {
        leaf_node_p = (static_cast<BwDeltaNode*>(leaf_node_p))->child_node;
      }
    } else if (isBasePage(leaf_node_p)) {
      // The last step is to search the key in the leaf, and we search
      // for the key in leaf page
      base_page_p = static_cast<BwLeafNode*>(leaf_node_p);
      return std::binary_search(base_page_p->data.begin(),
                                base_page_p->data.end(), key_less);
    } else {
      assert(false);
    }
  }  // while(1)

  // Should not reach here
  assert(false);
  return false;
}

/*
 * insert() - Insert a key-value pair into B-Tree
 *
 * NOTE: No duplicated key support
 */
template <typename KeyType, typename ValueType, typename KeyComparator>
bool BWTree<KeyType, ValueType, KeyComparator>::insert(const KeyType& key,
                                                       const ValueType& value) {
  bool insert_success;

  do {
    // First reach the leaf page where the key should be inserted
    PID page_pid = findLeafPage(key);

    // Then install an insertion record
    InstallDeltaResult result = installDeltaInsert(page_pid, key, value);
    if (result == install_need_consolidate) {
      insert_success = false;
      bool consolidation_success = consolidateLeafNode(page_pid);

      // If consolidation fails then we know some other thread
      // has performed consolidation for us
      // and we just try again
      if (consolidation_success == false) {
        // DON'T KNOW WHAT TO DO
        // TODO: ...
      }
    } else if (result == install_try_again) {
      // Don't need consolidate, but some other threads has
      // changed the delta chain
      insert_success = false;
    } else if (result == install_success) {
      // This branch should be the majority
      insert_success = true;
    } else {
      assert(false);
    }
  } while (insert_success == false);

  return true;
}

/*
 * erase() - Delete a key-value pair from the tree
 *
 * Since there could be duplicated keys, we need to
 * specify the data item locate the record for deletion
 */
template <typename KeyType, typename ValueType, typename KeyComparator>
bool BWTree<KeyType, ValueType, KeyComparator>::erase(const KeyType& key,
                                                      const ValueType& value) {
  bool delete_success;

  do {
    // First reach the leaf page where the key should be inserted
    PID page_pid = findLeafPage(key);

    // Then install an insertion record
    InstallDeltaResult result = installDeltaDelete(page_pid, key, value);
    if (result == install_need_consolidate) {
      delete_success = false;

      consolidateLeafNode(page_pid);
    } else if (result == install_try_again) {
      delete_success = false;
    } else if (result == install_success) {
      delete_success = true;
    } else {
      assert(false);
    }
  } while (delete_success == false);

  return true;
}

// Returns the first page where the key can reside
// For insert and delete this means the page on which delta record can be added
// For search means the first page the cursor needs to be constructed on
template <typename KeyType, typename ValueType, class KeyComparator>
typename BWTree<KeyType, ValueType, KeyComparator>::PID
BWTree<KeyType, ValueType, KeyComparator>::findLeafPage(const KeyType& key) {
  // Root should always have a valid pid
  assert(m_root != this->NONE_PID);
  PID curr_pid = m_root;
  BwNode* curr_node = mapping_table[curr_pid].load();

  PID parent_pid = this->NONE_PID;
  int deltaLeafLen = 0;
  int deltaIndexLen = 0;

  (void)deltaLeafLen;
  // Trigger consolidation

  // Trigger structure modifying operations
  // split, remove, merge

  while (1) {
    assert(curr_node != nullptr);
    if (curr_node->type == PageType::inner) {
      bwt_printf("Current page is inner\n");

      BwInnerNode* inner_node = static_cast<BwInnerNode*>(curr_node);
      // The consolidate has to ensure that it does not leave empty
      // inner node
      assert(inner_node->separators.size() > 0);

      if (deltaIndexLen > this->delta_chain_inner_thesh) {
        consolidateInnerNode(curr_pid);
        // Reset the chain length counter
        deltaIndexLen = 0;
        // Even if the consolidate fails or completes the search needs to
        // fetch the curr_node from the mapping table
        curr_node = mapping_table[curr_pid].load();
        continue;
      }

      if (inner_node->separators.size() < this->inner_node_size_min) {
        // Install a remove delta on top of the node
        installDeltaRemove(curr_pid);
        curr_node = mapping_table[curr_pid].load();
        continue;
      }

      if (inner_node->separators.size() > this->inner_node_size_max) {
        // Install a split delta on top of the node
        installDeltaSplit(curr_pid);
        curr_node = mapping_table[curr_pid].load();
        continue;
      }

      // TODO Change this to binary search
      bool found_sep = false;
      for (int i = 1; i < inner_node->separators.size(); i++) {
        bwt_printf("Inside for loop, i = %d\n", i);

        if (key_less(inner_node->separators[i].first, key)) {
          continue;
        } else {
          found_sep = true;
          // Change page
          parent_pid = curr_pid;
          deltaIndexLen = 0;
          // deltaLeafLen should be zero when you reach an inner node
          assert(deltaLeafLen == 0);
          curr_pid = inner_node->separators[i - 1].second;
          break;
        }
      }

      if (!found_sep) {
        // Change page
        parent_pid = curr_pid;
        deltaIndexLen = 0;
        // This should be zero when you reach an inner node
        assert(deltaLeafLen == 0);
        curr_pid = inner_node->separators.back().second;
      }

      curr_node = mapping_table[curr_pid].load();
      continue;

    } else if (curr_node->type == PageType::deltaIndexTermInsert) {
      BwDeltaIndexTermInsertNode* index_insert_delta =
          static_cast<BwDeltaIndexTermInsertNode*>(curr_node);
      // Increment the index delta chain counter
      deltaIndexLen++;
      if (deltaIndexLen > this->delta_chain_inner_thesh) {
        consolidateInnerNode(curr_pid);
        // Reset the chain length counter
        deltaIndexLen = 0;
        // Even if the consolidate fails or completes the search needs to
        // fetch the curr_node from the mapping table
        curr_node = mapping_table[curr_pid].load();
        continue;
      }
      if (key_greater(key, index_insert_delta->new_split_separator_key) &&
          key_lessequal(key, index_insert_delta->next_separator_key)) {
        // Change page
        parent_pid = curr_pid;
        deltaIndexLen = 0;
        // deltaLeafLen should be zero when you reach an inner node
        curr_pid = index_insert_delta->new_split_sibling;
        curr_node = mapping_table[curr_pid].load();
        continue;
      }

      curr_node = index_insert_delta->child_node;

    } else if (curr_node->type == PageType::deltaIndexTermDelete) {
      BwDeltaIndexTermDeleteNode* index_delete_delta =
          static_cast<BwDeltaIndexTermDeleteNode*>(curr_node);
      // Increment the index delta chain counter
      deltaIndexLen++;
      if (deltaIndexLen > this->delta_chain_inner_thesh) {
        consolidateInnerNode(curr_pid);
        // Reset the chain length counter
        deltaIndexLen = 0;
        // Even if the consolidate fails or completes the search needs to
        // fetch the curr_node from the mapping table
        curr_node = mapping_table[curr_pid].load();
        continue;
      }
      if (key_greater(key, index_delete_delta->merge_node_low_key) &&
          key_lessequal(key, index_delete_delta->next_separator_key)) {
        // Change page
        parent_pid = curr_pid;
        deltaIndexLen = 0;
        // deltaLeafLen should be zero when you reach an inner node
        curr_pid = index_delete_delta->node_to_merge_into;
        curr_node = mapping_table[curr_pid].load();
        continue;
      }

      curr_node = index_delete_delta->child_node;

    } else if (curr_node->type == PageType::deltaSplit) {
      // Our invariant is that there should be no delta chains on top of a
      // split node
      assert(deltaLeafLen == 0 && deltaIndexLen == 0);
      BwDeltaSplitNode* split_delta = static_cast<BwDeltaSplitNode*>(curr_node);

      // Install an IndexTermDeltaInsert and retry till it succeds. Thread
      // cannot proceed until this succeeds
      // 1) This might install multiple updates which say the same thing. The
      //    consolidate must be able to handle the duplicates
      // 2) The parent_pid might not be correct because the parent might have
      //    merged into some other node. This needs to be detected by the
      //    installIndexTermDeltaInsert and return the install_node_invalid
      //    which triggers a search from the root.
      InstallDeltaResult status = install_try_again;
      while (status != install_success) {
        status = installIndexTermDeltaInsert(parent_pid, split_delta);
        if (status == install_need_consolidate) {
          consolidateInnerNode(parent_pid);
        } else if (status == install_node_invalid) {
          // Restart from the top
          curr_pid = m_root;
          curr_node = mapping_table[curr_pid].load();
          continue;
        }
      }

      if (key_greater(key, split_delta->separator_key)) {
        // Change page
        parent_pid = curr_pid;
        // The delta chain lengths should be zero no need to reset them
        curr_pid = split_delta->split_sibling;
        curr_node = mapping_table[curr_pid].load();
        continue;
      }

      curr_node = split_delta->child_node;

    } else if (curr_node->type == PageType::deltaRemove) {
      // Note: should not trigger a remove on the left most leaf even if the
      // number of tuples is below a threshold

      // TODO: Install an installDeltaMerge on sibling and retry till it
      // succeds. Thread
      // cannot proceed until this succeeds
      assert(false);
    } else if (curr_node->type == PageType::deltaMerge) {
      // Our invariant is that there should be no delta chains on top of a
      // split node
      assert(deltaLeafLen == 0 && deltaIndexLen == 0);
      BwDeltaMergeNode* merge_delta = static_cast<BwDeltaMergeNode*>(curr_node);

      // Install an IndexTermDeltaDelete and retry till it succeds. Thread
      // cannot proceed until this succeeds
      // 1) This might install multiple updates which say the same thing. The
      //    consolidate must be able to handle the duplicates
      // 2) The parent_pid might not be correct because the parent might have
      //    merged into some other node. This needs to be detected by the
      //    installIndexTermDeltaDelete and return the install_node_invalid
      //    which triggers a search from the root.
      InstallDeltaResult status = install_try_again;
      while (status != install_success) {
        // status = installIndexTermDeltaDelete(parent_pid, split_delta);
        if (status == install_need_consolidate) {
          consolidateInnerNode(parent_pid);
        } else if (status == install_node_invalid) {
          // Restart from the top
          curr_pid = m_root;
          curr_node = mapping_table[curr_pid].load();
          continue;
        }
      }

      if (key_greater(key, merge_delta->separator_key)) {
        curr_node = merge_delta->merge_node;
      } else {
        curr_node = merge_delta->child_node;
      }
    } else if (curr_node->type == PageType::deltaInsert) {
      BwDeltaInsertNode* insert_node =
          static_cast<BwDeltaInsertNode*>(curr_node);
      if (key_lessequal(key, insert_node->ins_record.first)) {
        break;
      } else {
        curr_node = insert_node->child_node;
        assert(curr_node != nullptr);
      }
    } else if (curr_node->type == PageType::deltaDelete) {
      BwDeltaDeleteNode* delete_node =
          static_cast<BwDeltaDeleteNode*>(curr_node);
      if (key_lessequal(key, delete_node->del_record.first)) {
        break;
      } else {
        curr_node = delete_node->child_node;
        assert(curr_node != nullptr);
      }
    } else if (curr_node->type == PageType::leaf) {
      bwt_printf("Is a leaf node\n");

      BwLeafNode* leaf_node = static_cast<BwLeafNode*>(curr_node);

      bwt_printf("leaf_node_size = %lu\n", leaf_node->data.size());

      if (leaf_node->data.size() == 0 ||
          key_lessequal(key, leaf_node->data.back().first) ||
          leaf_node->next == this->NONE_PID) {
        bwt_printf(
            "key <= first in the leaf, or next leaf == NONE PID, Break!\n");

        break;
      } else {
        // Jump to sibling need to post an index update
        // There might need to consider duplicates separately
        curr_pid = leaf_node->next;
        curr_node = mapping_table[curr_pid].load();
      }
    } else {
      assert(false);
    }
  }  // while(1)

  return curr_pid;
}

template <typename KeyType, typename ValueType, class KeyComparator>
std::vector<ValueType> BWTree<KeyType, ValueType, KeyComparator>::find(
    const KeyType& key) {
  std::vector<ValueType> values;
  // Find the leaf page the key can possibly map into
  PID leaf_page = findLeafPage(key);
  assert(leaf_page != this->NONE_PID);

  BwNode* curr_node = mapping_table[leaf_page].load();

  // Check if the node is marked for consolidation, splitting or merging
  // BwNode* next_node = nullptr;
  while (curr_node != nullptr) {
    if (curr_node->type == PageType::leaf) {
    } else if (curr_node->type == PageType::deltaInsert) {
    } else if (curr_node->type == PageType::deltaDelete) {
    } else {
      assert(false);
    }
  }

  // Mark node for consolidation
  // Mark node for split
  // Mark node for merge
  return values;
}

template <typename KeyType, typename ValueType, class KeyComparator>
struct LessFn {
  LessFn(const KeyComparator& comp) : m_key_less(comp) {}

  bool operator()(const std::pair<KeyType, ValueType>& l,
                  const std::pair<KeyType, ValueType>& r) const {
    return m_key_less(std::get<0>(l), std::get<0>(r));
  }

  const KeyComparator& m_key_less;
};

template <typename KeyType, typename ValueType, class KeyComparator>
void BWTree<KeyType, ValueType, KeyComparator>::traverseAndConsolidateLeaf(
    BwNode* original_node, std::vector<BwNode*>& garbage_nodes,
    std::vector<std::pair<KeyType, ValueType>>& data, PID& sibling,
    bool& has_merge, BwNode*& merge_node, KeyType& lower_bound) {
  using LessFnT = LessFn<KeyType, ValueType, KeyComparator>;

  LessFnT less_fn(m_key_less);
  std::set<std::pair<KeyType, ValueType>, LessFnT> insert_records(less_fn);
  std::set<std::pair<KeyType, ValueType>, LessFnT> delete_records(less_fn);

  bool has_split = false;
  KeyType split_separator_key = KeyType();
  PID new_sibling;

  has_merge = false;
  KeyType merge_separator_key = KeyType();
  merge_node = nullptr;

  BwNode* node = original_node;
  while (node->type != leaf) {
    switch (node->type) {
      case deltaInsert: {
        BwDeltaInsertNode* insert_node = static_cast<BwDeltaInsertNode*>(node);
        // If we have a delete for this record, don't add
        auto it = delete_records.find(insert_node->ins_record);
        if (it != delete_records.end()) {
          // Have existing delete record, get rid of it
          delete_records.erase(it);
        } else {
          insert_records.insert(insert_node->ins_record);
        }
        break;
      }
      case deltaDelete: {
        BwDeltaDeleteNode* delete_node = static_cast<BwDeltaDeleteNode*>(node);
        delete_records.insert(delete_node->del_record);
        break;
      }
      case deltaSplit: {
        // Our invariant is that split nodes always force a consolidate, so
        // should be at the top
        assert(node == original_node);  // Ensure the split is at the top
        assert(!has_split);             // Ensure this is the only split
        BwDeltaSplitNode* split_node = static_cast<BwDeltaSplitNode*>(node);
        has_split = true;
        split_separator_key = split_node->separator_key;
        new_sibling = split_node->split_sibling;
        break;
      }
      case deltaMerge: {
        // Same as split, invariant is that merge nodes always force a
        // consolidate, so should be at the top
        assert(node == original_node);
        BwDeltaMergeNode* merge_delta = static_cast<BwDeltaMergeNode*>(node);
        has_merge = true;
        merge_separator_key = merge_delta->separator_key;
        merge_node = merge_delta->merge_node;
        break;
      }
      default:
        assert(false);
    }

    garbage_nodes.push_back(node);
    node = static_cast<BwDeltaNode*>(node)->child_node;
    assert(node != nullptr);
  }

  // node is a leaf node
  BwLeafNode* leaf_node = static_cast<BwLeafNode*>(node);

  lower_bound = leaf_node->lower_bound;
  if (has_split) {
    // Change sibling pointer if we did a split
    sibling = new_sibling;
  } else {
    sibling = leaf_node->next;
  }

  for (const std::pair<KeyType, ValueType>& tuple : leaf_node->data) {
    auto it = delete_records.find(tuple);
    if (it != delete_records.end()) {
      // Deleting to ensure correctness but not necessary
      delete_records.erase(it);
    } else {
      // Either not split or is split and tuple is less than separator
      if (!has_split || key_less(std::get<0>(tuple), split_separator_key)) {
        data.push_back(tuple);
      }
    }
  }
  // Should have deleted all the records
  // Even with split, this should be fine because all the tuples will still
  // be in the original base page
  assert(delete_records.empty());

  // Insert only new records before the split
  for (const std::pair<KeyType, ValueType>& tuple : insert_records) {
    if (!has_split || key_less(std::get<0>(tuple), split_separator_key)) {
      data.push_back(tuple);
    }
  }
  // This is not very efficient, but ok for now
  std::sort(data.begin(), data.end(), less_fn);
}

template <typename KeyType, typename ValueType, class KeyComparator>
bool BWTree<KeyType, ValueType, KeyComparator>::consolidateLeafNode(PID id) {
  // Keep track of nodes so we can garbage collect later
  BwNode* original_node = mapping_table[id].load();

  std::vector<BwNode*> garbage_nodes;
  std::vector<std::pair<KeyType, ValueType>> data;
  PID sibling;
  bool has_merge = false;
  BwNode* merge_node = nullptr;
  KeyType lower_bound;
  traverseAndConsolidateLeaf(original_node, garbage_nodes, data, sibling,
                             has_merge, merge_node, lower_bound);
  if (has_merge) {
    BwNode* dummy_node;
    KeyType dummy_bound;
    traverseAndConsolidateLeaf(merge_node, garbage_nodes, data, sibling,
                               has_merge, dummy_node, dummy_bound);
    assert(!has_merge);
  }

  BwLeafNode* consolidated_node = new BwLeafNode(lower_bound, sibling);
  consolidated_node->data.swap(data);

  bool result = mapping_table[id].compare_exchange_strong(original_node,
                                                          consolidated_node);
  if (!result) {
    // Failed, cleanup
    delete consolidated_node;
  } else {
    // Succeeded, request garbage collection of processed nodes
  }
  return result;
}

template <typename KeyType, typename ValueType, class KeyComparator>
void BWTree<KeyType, ValueType, KeyComparator>::traverseAndConsolidateInner(
    BwNode* original_node, std::vector<BwNode*>& garbage_nodes,
    std::vector<std::pair<KeyType, PID>>& separators, bool& has_merge,
    BwNode*& merge_node, KeyType& lower_bound) {
  std::vector<std::pair<KeyType, PID>> insert_separators;
  std::vector<std::pair<KeyType, PID>> delete_separators;

  // Split variables
  bool has_split = false;
  KeyType split_separator_key = KeyType();

  // Merge variables
  has_merge = false;
  KeyType merge_separator_key = KeyType();
  merge_node = nullptr;

  // Keep track of nodes so we can garbage collect later
  // NOTE(apoms): This logic is almost identical to the leaf node traversal
  // but its more efficient because it was my second time writing it
  BwNode* node = original_node;
  while (node->type != inner) {
    switch (node->type) {
      case deltaIndexTermInsert: {
        BwDeltaIndexTermInsertNode* insert_node =
            static_cast<BwDeltaIndexTermInsertNode*>(node);
        if (!has_split || key_less(insert_node->new_split_separator_key,
                                   split_separator_key)) {
          insert_separators.push_back({insert_node->new_split_separator_key,
                                       insert_node->new_split_sibling});
        }
        break;
      }
      case deltaIndexTermDelete: {
        BwDeltaIndexTermDeleteNode* delete_node =
            static_cast<BwDeltaIndexTermDeleteNode*>(node);
        if (!has_split ||
            key_less(delete_node->remove_node_low_key, split_separator_key)) {
          delete_separators.push_back(
              {delete_node->remove_node_low_key, delete_node->node_to_remove});
        }
        break;
      }
      case deltaSplit: {
        // Our invariant is that split nodes always force a consolidate, so
        // should be at the top
        assert(node == original_node);  // Ensure the split is at the top
        assert(!has_split);             // Ensure this is the only split
        BwDeltaSplitNode* split_node = static_cast<BwDeltaSplitNode*>(node);
        has_split = true;
        split_separator_key = split_node->separator_key;
        break;
      }
      case deltaMerge: {
        // Same as split, invariant is that merge nodes always force a
        // consolidate, so should be at the top
        assert(node == original_node);
        BwDeltaMergeNode* merge_delta = static_cast<BwDeltaMergeNode*>(node);
        has_merge = true;
        merge_separator_key = merge_delta->separator_key;
        merge_node = merge_delta->merge_node;
        break;
      }
      default:
        assert(false);
    }

    garbage_nodes.push_back(node);
    node = static_cast<BwDeltaNode*>(node)->child_node;
  }

  BwInnerNode* inner_node = static_cast<BwInnerNode*>(node);

  lower_bound = inner_node->lower_bound;

  typename std::vector<std::pair<KeyType, PID>>::iterator base_end;
  if (has_split) {
    // Find end of separators if split
    base_end =
        std::lower_bound(inner_node->separators.begin(),
                         inner_node->separators.end(), split_separator_key,
                         [=](const std::pair<KeyType, PID>& l, const KeyType& r)
                             -> bool { return m_key_less(std::get<0>(l), r); });
  } else {
    base_end = inner_node->separators.end();
  }

  // Merge with difference
  auto less_fn =
      [=](const std::pair<KeyType, PID>& l, const std::pair<KeyType, PID>& r)
          -> bool { return m_key_less(std::get<0>(l), std::get<0>(r)); };

  // Separators might have sorted data in it already (e.g. if there was a merge
  // node and so the left half has already been consolidated into separators).
  // Thus we keep track of the end so we know which part we need to make sure is
  // sorted (all the data in separators should be less than any data in here
  // because it was from a left sibling node).
  auto separators_start = separators.end();
  // Not very efficient (difference and merge could be combined)
  std::set_difference(inner_node->separators.begin(), base_end,
                      delete_separators.begin(), delete_separators.end(),
                      std::inserter(separators, separators.end()), less_fn);
  // Append sorted inserts to end
  auto middle_it = separators.end();
  separators.insert(separators.end(), insert_separators.begin(),
                    insert_separators.end());
  // Merge the results together
  std::inplace_merge(separators_start, middle_it, separators.end(), less_fn);
}

template <typename KeyType, typename ValueType, class KeyComparator>
bool BWTree<KeyType, ValueType, KeyComparator>::consolidateInnerNode(PID id) {
  BwNode* original_node = mapping_table[id].load();

  // Keep track of nodes so we can garbage collect later
  std::vector<BwNode*> garbage_nodes;
  std::vector<std::pair<KeyType, PID>> separators;
  bool has_merge = false;
  BwNode* merge_node = nullptr;
  KeyType lower_bound;
  traverseAndConsolidateInner(original_node, garbage_nodes, separators,
                              has_merge, merge_node, lower_bound);
  if (has_merge) {
    BwNode* dummy_node;
    KeyType dummy_bound;
    traverseAndConsolidateInner(merge_node, garbage_nodes, separators,
                                has_merge, dummy_node, dummy_bound);
    assert(!has_merge);
  }

  BwInnerNode* consolidated_node = new BwInnerNode(lower_bound);
  consolidated_node->separators.swap(separators);

  bool result = mapping_table[id].compare_exchange_strong(original_node,
                                                          consolidated_node);
  if (!result) {
    // Failed, cleanup
    delete consolidated_node;
  } else {
    // Succeeded, request garbage collection of processed nodes
  }
  return true;
}

// This function will assign a page ID for a given page, and put that page into
// the mapping table
//
// NOTE: This implementation referred to the Bw-Tree implementation on github:
// >> https://github.com/flode/BwTree/blob/master/bwtree.hpp
template <typename KeyType, typename ValueType, class KeyComparator>
typename BWTree<KeyType, ValueType, KeyComparator>::PID
BWTree<KeyType, ValueType, KeyComparator>::installPage(BwNode* new_node_p) {
  // Let's assume first this will not happen; If it happens
  // then we change this to a DEBUG output
  // Need to use a variable length data structure
  assert(next_pid < max_table_size);

  // Though it is operating on std::atomic<PID>, the ++ operation
  // will be reflected to the underlying storage
  // Also threads will be serialized here to get their own PID
  // Once a PID is assigned, different pages on different slots will
  // interfere with each other
  PID assigned_pid = next_pid++;
  mapping_table[assigned_pid] = new_node_p;

  current_mapping_table_size++;

  return assigned_pid;
}

template <typename KeyType, typename ValueType, typename KeyComparator>
typename BWTree<KeyType, ValueType, KeyComparator>::InstallDeltaResult
BWTree<KeyType, ValueType, KeyComparator>::installDeltaInsert(
    PID leaf_pid, const KeyType& key, const ValueType& value) {
  bool cas_success;
  auto ins_record = std::pair<KeyType, ValueType>(key, value);

  BwNode* old_leaf_p = mapping_table[leaf_pid].load();

  if (old_leaf_p->type == PageType::deltaMerge ||
      old_leaf_p->type == PageType::deltaRemove ||
      old_leaf_p->type == PageType::deltaSplit) {
    return install_need_consolidate;
  }

  BwNode* new_leaf_p = (BwNode*)new BwDeltaInsertNode(old_leaf_p, ins_record);

  // Either the page has been consolidated, in which case we try again,
  // or the page has been appended another record, in which case we also
  // try again
  cas_success =
      mapping_table[leaf_pid].compare_exchange_strong(old_leaf_p, new_leaf_p);
  if (cas_success == false) {
    delete new_leaf_p;
    return install_try_again;
  }

  return install_success;
}

template <typename KeyType, typename ValueType, typename KeyComparator>
typename BWTree<KeyType, ValueType, KeyComparator>::InstallDeltaResult
BWTree<KeyType, ValueType, KeyComparator>::installDeltaDelete(
    PID leaf_pid, const KeyType& key, const ValueType& value) {
  bool cas_success;
  auto delete_record = std::pair<KeyType, ValueType>(key, value);

  BwNode* old_leaf_p = mapping_table[leaf_pid].load();

  if (old_leaf_p->type == PageType::deltaMerge ||
      old_leaf_p->type == PageType::deltaRemove ||
      old_leaf_p->type == PageType::deltaSplit) {
    return install_need_consolidate;
  }

  BwNode* new_leaf_p =
      (BwNode*)new BwDeltaDeleteNode(old_leaf_p, delete_record);

  cas_success =
      mapping_table[leaf_pid].compare_exchange_strong(old_leaf_p, new_leaf_p);
  if (cas_success == false) {
    delete new_leaf_p;
    return install_try_again;
  }

  return install_success;
}

template <typename KeyType, typename ValueType, typename KeyComparator>
typename BWTree<KeyType, ValueType, KeyComparator>::InstallDeltaResult
BWTree<KeyType, ValueType, KeyComparator>::installIndexTermDeltaInsert(
    __attribute__((unused)) PID node,
    __attribute__((unused)) const BwDeltaSplitNode* split_node) {
  return install_try_again;
}

template <typename KeyType, typename ValueType, typename KeyComparator>
typename BWTree<KeyType, ValueType, KeyComparator>::InstallDeltaResult
BWTree<KeyType, ValueType, KeyComparator>::installIndexTermDeltaDelete(
    __attribute__((unused)) PID node,
    __attribute__((unused)) const KeyType& key,
    __attribute__((unused)) const ValueType& value) {
  return install_try_again;
}

template <typename KeyType, typename ValueType, typename KeyComparator>
typename BWTree<KeyType, ValueType, KeyComparator>::InstallDeltaResult
BWTree<KeyType, ValueType, KeyComparator>::installDeltaRemove(
    __attribute__((unused)) PID node) {
  return install_try_again;
}

template <typename KeyType, typename ValueType, typename KeyComparator>
typename BWTree<KeyType, ValueType, KeyComparator>::InstallDeltaResult
BWTree<KeyType, ValueType, KeyComparator>::installDeltaMerge(
    __attribute__((unused)) PID node, __attribute__((unused)) PID sibling) {
  return install_try_again;
}

template <typename KeyType, typename ValueType, typename KeyComparator>
typename BWTree<KeyType, ValueType, KeyComparator>::InstallDeltaResult
BWTree<KeyType, ValueType, KeyComparator>::installDeltaSplit(
    __attribute__((unused)) PID node) {
  return install_try_again;
}

/*
 * splitLeafNode() - Find a pivot and post deltaSplit record both the leaf node
 */
template <typename KeyType, typename ValueType, typename KeyComparator>
bool BWTree<KeyType, ValueType, KeyComparator>::splitLeafNode(
    __attribute__((unused)) PID id) {
  return false;
}

template <typename KeyType, typename ValueType, typename KeyComparator>
bool BWTree<KeyType, ValueType, KeyComparator>::splitInnerNode(
    __attribute__((unused)) PID id) {
  // Has to handle the root node splitting and adding another level in
  // the tree
  return false;
}

template <typename KeyType, typename ValueType, typename KeyComparator>
bool BWTree<KeyType, ValueType, KeyComparator>::mergeLeafNode(
    __attribute__((unused)) PID id) {
  return false;
}

template <typename KeyType, typename ValueType, typename KeyComparator>
bool BWTree<KeyType, ValueType, KeyComparator>::mergeInnerNode(
    __attribute__((unused)) PID id) {
  return false;
}

}  // End index namespace
}  // End peloton namespace

