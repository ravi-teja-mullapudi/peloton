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

#define BWTREE_DEBUG

#ifdef BWTREE_DEBUG

#define bwt_printf(fmt, ...) do { printf("%-24s(): " fmt, __FUNCTION__, ## __VA_ARGS__); }while(0);
#define bwt_printf_red(fmt, ...) do { printf("\033[1;31m%-24s(): " fmt "\033[0m", \
                                             __FUNCTION__, ## __VA_ARGS__); }while(0);
#define bwt_printf_redb(fmt, ...) do { printf("\033[1;41m%-24s(): " fmt "\033[0m", \
                                              __FUNCTION__, ## __VA_ARGS__); }while(0);
#define bwt_printf_green(fmt, ...) do { printf("\033[1;32m%-24s(): " fmt "\033[0m", \
                                               __FUNCTION__, ## __VA_ARGS__); }while(0);
#define bwt_printf_greenb(fmt, ...) do { printf("\033[1;42m%-24s(): " fmt "\033[0m", \
                                                __FUNCTION__, ## __VA_ARGS__); }while(0);
#define bwt_printf_blue(fmt, ...) do { printf("\033[1;34m%-24s(): " fmt "\033[0m", \
                                              __FUNCTION__, ## __VA_ARGS__); }while(0);
#define bwt_printf_blueb(fmt, ...) do { printf("\033[1;44m%-24s(): " fmt "\033[0m", \
                                               __FUNCTION__, ## __VA_ARGS__); }while(0);
#else

#define bwt_printf(args...) do {}while(0);
#define bwt_printf_red(fmt, ...) do {}while(0);
#define bwt_printf_redb(fmt, ...) do {}while(0);
#define bwt_printf_green(fmt, ...) do {}while(0);
#define bwt_printf_greenb(fmt, ...) do {}while(0);
#define bwt_printf_blue(fmt, ...) do {}while(0);
#define bwt_printf_blueb(fmt, ...) do {}while(0);

#endif

namespace peloton {
namespace index {

// Look up the stx btree interface for background.
// peloton/third_party/stx/btree.h
template <typename KeyType, typename ValueType, typename KeyComparator>
class BWTree {
 private:
  using PID = uint32_t;

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
    // Inner type
    deltaSplit,
    deltaIndexTermInsert,
    deltaIndexTermDelete,
    deltaRemove,
    deltaMerge,
  };

  enum InstallDeltaResult {
    install_success,
    install_try_again,
    install_need_consolidate,
  };

  /*
   * class BwNode - Generic node class; inherited by leaf, inner
   * and delta node
   */
  class BwNode {
   public:
    BwNode(PageType _type, KeyComparator _m_key_less)
        : m_key_less(_m_key_less), type(_type) {}

    const KeyComparator& m_key_less;
    PageType type;

    inline bool key_less(const KeyType& a, const KeyType& b) const {
      return m_key_less(a, b);
    }

    inline bool key_lessequal(const KeyType& a, const KeyType& b) const {
      return !m_key_less(b, a);
    }

    inline bool key_greater(const KeyType& a, const KeyType& b) const {
      return m_key_less(b, a);
    }

    inline bool key_greaterequal(const KeyType& a, const KeyType& b) const {
      return !m_key_less(a, b);
    }

    inline bool key_equal(const KeyType& a, const KeyType& b) const {
      return !m_key_less(a, b) && !m_key_less(b, a);
    }

  };  // class BwNode

  /*
   * class BwDeltaNode - Delta page
   */
  class BwDeltaNode : public BwNode {
   public:
    BwDeltaNode(PageType _type, BwNode* _child_node, KeyComparator _m_key_less)
        : BwNode(_type, _m_key_less) {
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
                      std::pair<KeyType, ValueType> _ins_record,
                      KeyComparator _m_key_less)
        : BwDeltaNode(PageType::deltaInsert, _child_node, _m_key_less) {
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
                      std::pair<KeyType, ValueType> _del_record,
                      KeyComparator _m_key_less)
        : BwDeltaNode(PageType::deltaDelete, _child_node, _m_key_less) {
      del_record = _del_record;
    }

    std::pair<KeyType, ValueType> del_record;
  };  // class BwDeltaDeleteNode

  /*
   * class BwDeltaSplitNode - Leaf and inner split node
   */
  class BwDeltaSplitNode : public BwDeltaNode {
   public:
    BwDeltaSplitNode(BwNode* _child_node, KeyType separator, PID split_sibling,
                     KeyComparator _m_key_less)
        : BwDeltaNode(PageType::deltaSplit, _child_node, _m_key_less),
          separator_key(separator),
          split_sibling(split_sibling) {}

    KeyType separator_key;
    PID split_sibling;
  };  // class BwDeltaSplitNode

  /*
   * class BwDeltaIndexTermInsertNode - Index separator add
   */
  class BwDeltaIndexTermInsertNode : public BwDeltaNode {
   public:
    BwDeltaIndexTermInsertNode(BwNode* _child_node,
                               KeyType new_split_separator_key,
                               PID new_split_sibling,
                               KeyType next_separator_key,
                               KeyComparator _m_key_less)
        : BwDeltaNode(PageType::deltaIndexTermInsert, _child_node, _m_key_less),
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
                               KeyType remove_node_high_key,
                               KeyComparator _m_key_less)
        : BwDeltaNode(PageType::deltaIndexTermDelete, _child_node, _m_key_less),
          node_to_merge_into(node_to_merge_into),
          node_to_remove(node_to_remove),
          merge_node_low_key(merge_node_low_key),
          remove_node_high_key(remove_node_high_key) {}

    PID node_to_merge_into;
    PID node_to_remove;
    KeyType merge_node_low_key;
    KeyType remove_node_low_key;
    KeyType remove_node_high_key;
  };  // class BwDeltaIndexTermDeleteNode

  /*
   * class BwDeltaRemoveNode - Delete and free page
   *
   * NOTE: This is not delete key node
   */
  class BwDeltaRemoveNode : public BwDeltaNode {
   public:
    BwDeltaRemoveNode(BwNode* _child_node, KeyComparator _m_key_less)
        : BwDeltaNode(PageType::deltaRemove, _child_node, _m_key_less) {}
  };  // class BwDeltaRemoveNode

  /*
   * class BwDeltaMergeNode - Merge two pages into one
   */
  class BwDeltaMergeNode : public BwDeltaNode {
   public:
    BwDeltaMergeNode(BwNode* _child_node, KeyType separator_key,
                     BwNode* merge_node, KeyComparator _m_key_less)
        : BwDeltaNode(PageType::deltaMerge, _child_node, _m_key_less),
          separator_key(separator_key),
          merge_node(merge_node) {}

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
    BwInnerNode(KeyType _lower_bound, PID _next, KeyComparator _m_key_less)
        : BwNode(PageType::inner, _m_key_less),
          lower_bound(_lower_bound),
          next(_next) {}

    const KeyType lower_bound;
    const PID next;
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
    BwLeafNode(KeyType _lower_bound, PID _next, KeyComparator _m_key_less)
        : BwNode(PageType::leaf, _m_key_less),
          lower_bound(_lower_bound),
          next(_next) {}

    // TODO : maybe we need to implement both a left and right pointer for
    // now sticking with just next
    bool comp_data(const std::pair<KeyType, ValueType>& d1,
                   const std::pair<KeyType, ValueType>& d2) {
      return key_less(d1.first, d2.first);
    }

    // Check if a key exists in the node
    bool find(const KeyType& key) {
      bwt_printf("LeafNode.find(%d)\n", key);
      for (int i = 0; i < data.size(); i++) {
        if (key_equal(data[i].first, key)) return true;
      }

      bwt_printf("LeafNode.find() returns false\n");
      return false;

      // Uncomment this in the future
      // return std::binary_search(data.begin(), data.end(), key, comp_data);
    }

    const KeyType lower_bound;
    const PID next;

    // Elastic container to allow for separation of consolidation, splitting
    // and merging
    std::vector<std::pair<KeyType, ValueType>> data;
  };  // class BwLeafNode

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
      std::vector<std::pair<KeyType, PID>>& separators, PID& sibling,
      bool& has_merge, BwNode*& merge_node, KeyType& lower_bound);

  bool consolidateInnerNode(PID id);

  bool isLeaf(BwNode* n);
  bool isLeafPID(PID pid);

  // Below three methods are inline helper functions
  // to help identification of different type of leaf nodes
  bool isDeltaInsert(BwNode* node_p) { return node_p->type == deltaInsert; }

  bool isDeltaDelete(BwNode* node_p) { return node_p->type == deltaDelete; }

  bool isBasePage(BwNode* node_p) { return node_p->type == leaf; }

  PID findLeafPage(const KeyType& key);

  void splitIndexNode(void);

  void splitLeafNode(void);

  void mergeInnerNode(void);

  void mergeLeafNode(void);

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
  BwLeafNode* initial_leaf = new BwLeafNode(KeyType(), NONE_PID, m_key_less);

  PID leaf_pid = installPage(initial_leaf);
  BwInnerNode* initial_inner = new BwInnerNode(KeyType(), NONE_PID, m_key_less);

  PID inner_pid = installPage(initial_inner);
  initial_inner->separators.push_back({KeyType(), leaf_pid});

  m_root = inner_pid;

  bwt_printf("Init: Initializer returns. Leaf = %u, inner = %u\n", leaf_pid,
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
 * isLeaf() - Returns true if the BwNode * refers to a leaf page
 * or its delta page
 *
 * NOTE & TODO: Currently this function is incorrect, as we need to traverse
 * the delta chain to the base page because of ambiguity caused by deltasplit,
 * delta merge and deltaremove
 */
template <typename KeyType, typename ValueType, typename KeyComparator>
bool BWTree<KeyType, ValueType, KeyComparator>::isLeaf(BwNode* n) {
  bool is_leaf = false;
  switch (n->type) {
    case deltaDelete:
    case deltaInsert:
    case leaf:
      is_leaf = true;
    case deltaSplit:
    case deltaIndexTermInsert:
    case deltaIndexTermDelete:
    case inner:
      break;
    default:
      assert(false);
  }

  return is_leaf;
}

/*
 * isLeafPID() - Returns true if a PID refers to a leaf node
 *
 * It acts as a wrapper to isLeaf(). Please note that even
 * if the PID-pointer relation has changed, identity of leaf
 * will not change
 */
template <typename KeyType, typename ValueType, class KeyComparator>
bool BWTree<KeyType, ValueType, KeyComparator>::isLeafPID(PID pid) {
  return isLeaf(mapping_table[pid].load());
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
  assert(isLeafPID(page_pid));

  BwNode* leaf_node_p = mapping_table[page_pid].load();
  assert(leaf_node_p != nullptr);

  BwDeltaInsertNode* insert_page_p = nullptr;
  BwDeltaDeleteNode* delete_page_p = nullptr;
  BwLeafNode* base_page_p = nullptr;
  std::pair<KeyType, ValueType>* pair_p = nullptr;

  while (1) {
    if (isDeltaInsert(leaf_node_p)) {
      insert_page_p = static_cast<BwDeltaInsertNode*>(leaf_node_p);

      bwt_printf("See DeltaInsert Page: %d\n",
             insert_page_p->ins_record.first);

      // If we see an insert node first, then this implies that the
      // key does exist in the future comsolidated version of the page
      if (key_equal(insert_page_p->ins_record.first, key) == true) {
        return true;
      } else {
        leaf_node_p = (static_cast<BwDeltaNode*>(leaf_node_p))->child_node;
      }
    } else if (isDeltaDelete(leaf_node_p)) {
      delete_page_p = static_cast<BwDeltaDeleteNode*>(leaf_node_p);

      // For delete record it implies the node has been removed
      if (key_equal(delete_page_p->del_record.first, key) == true) {
        return false;
      } else {
        leaf_node_p = (static_cast<BwDeltaNode*>(leaf_node_p))->child_node;
      }
    } else if (isBasePage(leaf_node_p)) {
      // The last step is to search the key in the leaf, and we search
      // for the key in leaf page
      // TODO: Add support for duplicated key
      base_page_p = static_cast<BwLeafNode*>(leaf_node_p);
      return base_page_p->find(key);
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
    assert(isLeafPID(page_pid));

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
    assert(isLeafPID(page_pid));

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

  // Zig-zag traversal
  while (1) {
    assert(curr_node != nullptr);
    if (curr_node->type == PageType::inner) {
      bwt_printf("Current page is inner\n");

      BwInnerNode* inner_node = static_cast<BwInnerNode*>(curr_node);
      assert(inner_node->separators.size() > 0);

      // TODO Change this to binary search
      bool found_sep = false;
      for (int i = 1; i < inner_node->separators.size(); i++) {
        bwt_printf("Inside for loop, i = %d\n", i);

        if (key_less(inner_node->separators[i - 1].first, key) &&
            key_less(inner_node->separators[i].first, key)) {
          continue;
        } else {
          found_sep = true;
          curr_pid = inner_node->separators[i - 1].second;
          break;
        }
      }

      if (!found_sep) {
        bwt_printf("Did not find sep\n");

        if (inner_node->next == this->NONE_PID) {
          bwt_printf("inner_node->next == NONE PID\n");

          curr_pid = inner_node->separators.back().second;
        } else {
          // Jump to sibling need to post an index update
          // There might need to consider duplicates separately
          curr_pid = inner_node->next;
        }
      }

      curr_node = mapping_table[curr_pid].load();
      continue;

    } else if (curr_node->type == PageType::deltaIndexTermInsert) {
      BwDeltaIndexTermInsertNode* index_insert_delta =
          static_cast<BwDeltaIndexTermInsertNode*>(curr_node);
      if (key_greater(key, index_insert_delta->new_split_separator_key) &&
          key_lessequal(key, index_insert_delta->next_separator_key)) {
        curr_pid = index_insert_delta->new_split_sibling;
        curr_node = mapping_table[curr_pid].load();
        continue;
      }

      curr_node = index_insert_delta->child_node;

    } else if (curr_node->type == PageType::deltaIndexTermDelete) {
      BwDeltaIndexTermDeleteNode* index_delete_delta =
          static_cast<BwDeltaIndexTermDeleteNode*>(curr_node);
      if (key_greater(key, index_delete_delta->merge_node_low_key) &&
          key_lessequal(key, index_delete_delta->remove_node_high_key)) {
        curr_pid = index_delete_delta->node_to_merge_into;
        curr_node = mapping_table[curr_pid].load();
        continue;
      }

      curr_node = index_delete_delta->child_node;

    } else if (curr_node->type == PageType::deltaSplit) {
      BwDeltaSplitNode* split_delta = static_cast<BwDeltaSplitNode*>(curr_node);
      if (key_greater(key, split_delta->separator_key)) {
        curr_pid = split_delta->split_sibling;
        curr_node = mapping_table[curr_pid].load();
        continue;
      }

      curr_node = split_delta->child_node;

    } else if (curr_node->type == PageType::deltaRemove) {
      // Have to find the left sibling
      assert(false);
    } else if (curr_node->type == PageType::deltaMerge) {
      BwDeltaMergeNode* merge_delta = static_cast<BwDeltaMergeNode*>(curr_node);
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
        bwt_printf("key <= first in the leaf, or next leaf == NONE PID, Break!\n");

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

/*
 * splitLeafNode() - Find a pivot and post delta record both on
 * the leaf node, and on its parent node
 */
template <typename KeyType, typename ValueType, class KeyComparator>
void BWTree<KeyType, ValueType, KeyComparator>::splitLeafNode(void) {}

template <typename KeyType, typename ValueType, class KeyComparator>
std::vector<ValueType> BWTree<KeyType, ValueType, KeyComparator>::find(
    const KeyType& key) {
  std::vector<ValueType> values;
  // Find the leaf page the key can possibly map into
  PID leaf_page = findLeafPage(key);
  assert(leaf_page != this->NONE_PID);

  BwNode* curr_node = mapping_table[leaf_page].load();
  // the node should be a leaf node
  assert(isLeaf(curr_node));

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
  KeyType split_separator_key;
  PID new_sibling;

  has_merge = false;
  KeyType merge_separator_key;
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

  BwLeafNode* consolidated_node =
      new BwLeafNode(lower_bound, sibling, m_key_less);
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
    std::vector<std::pair<KeyType, PID>>& separators, PID& sibling,
    bool& has_merge, BwNode*& merge_node, KeyType& lower_bound) {
  std::vector<std::pair<KeyType, PID>> insert_separators;
  std::vector<std::pair<KeyType, PID>> delete_separators;

  // Split variables
  bool has_split = false;
  KeyType split_separator_key;
  PID new_sibling;

  // Merge variables
  has_merge = false;
  KeyType merge_separator_key;
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
    sibling = new_sibling;
  } else {
    base_end = inner_node->separators.end();
    sibling = inner_node->next;
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
  auto middle_it = separators.insert(
      separators.end(), insert_separators.begin(), insert_separators.end());
  // Merge the results together
  std::inplace_merge(separators_start, middle_it, separators.end());
}

template <typename KeyType, typename ValueType, class KeyComparator>
bool BWTree<KeyType, ValueType, KeyComparator>::consolidateInnerNode(PID id) {
  BwNode* original_node = mapping_table[id].load();

  // Keep track of nodes so we can garbage collect later
  std::vector<BwNode*> garbage_nodes;
  std::vector<std::pair<KeyType, PID>> separators;
  PID sibling;
  bool has_merge = false;
  BwNode* merge_node = nullptr;
  KeyType lower_bound;
  traverseAndConsolidateInner(original_node, garbage_nodes, separators, sibling,
                              has_merge, merge_node, lower_bound);
  if (has_merge) {
    BwNode* dummy_node;
    KeyType dummy_bound;
    traverseAndConsolidateInner(merge_node, garbage_nodes, separators, sibling,
                                has_merge, dummy_node, dummy_bound);
    assert(!has_merge);
  }

  BwInnerNode* consolidated_node =
      new BwInnerNode(lower_bound, sibling, m_key_less);
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

  // We must be working on a leaf page
  // This includes base page, delete page and insert page
  assert(isLeaf(old_leaf_p));

  BwNode* new_leaf_p =
      (BwNode*)new BwDeltaInsertNode(old_leaf_p, ins_record, m_key_less);

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

  assert(isLeaf(old_leaf_p));

  BwNode* new_leaf_p =
      (BwNode*)new BwDeltaDeleteNode(old_leaf_p, delete_record, m_key_less);

  cas_success =
      mapping_table[leaf_pid].compare_exchange_strong(old_leaf_p, new_leaf_p);
  if (cas_success == false) {
    delete new_leaf_p;
    return install_try_again;
  }

  return install_success;
}

}  // End index namespace
}  // End peloton namespace

