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
#include <stack>
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

  //////////////////////////////////////////////////////////////////////////////
  /// Performance configuration constants
  constexpr static unsigned int max_table_size = 1 << 24;
  // Threshold of delta chain length on an inner node to trigger a consolidate
  constexpr static unsigned int DELTA_CHAIN_INNER_THRESHOLD = 8;
  // Threshold of delta chain length on a leaf node to trigger a consolidate
  constexpr static unsigned int DELTA_CHAIN_LEAF_THESH = 8;
  // Node sizes for triggering splits and merges on inner nodes
  constexpr static unsigned int INNER_NODE_SIZE_MIN = 8;
  constexpr static unsigned int INNER_NODE_SIZE_MAX = 16;
  // Node sizes for triggering splits and merges on leaf nodes
  constexpr static unsigned int LEAF_NODE_SIZE_MIN = 8;
  constexpr static unsigned int LEAF_NODE_SIZE_MAX = 16;
  // Debug constant: The maximum number of iterations we could do
  // It prevents dead loop hopefully
  constexpr static int ITER_MAX = 99999;

  // Enumeration of the types of nodes required in updating both the values
  // and the index in the BW Tree. Currently only adding node types for
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
   * class BWNode - Generic node class; inherited by leaf, inner
   * and delta node
   */
  class BWNode {
   public:
    BWNode(PageType _type) : type(_type) {}
    PageType type;
  };  // class BWNode

  /*
   * class BWDeltaNode - Delta page
   */
  class BWDeltaNode : public BWNode {
   public:
    BWDeltaNode(PageType _type, BWNode* _child_node) : BWNode(_type) {
      child_node = _child_node;
    }

    BWNode* child_node;
  };  // class BWDeltaNode

  /*
   * class BWDeltaInsertNode - Key insert delta
   */
  class BWDeltaInsertNode : public BWDeltaNode {
   public:
    BWDeltaInsertNode(BWNode* _child_node,
                      std::pair<KeyType, ValueType> _ins_record)
        : BWDeltaNode(PageType::deltaInsert, _child_node) {
      ins_record = _ins_record;
    }

    std::pair<KeyType, ValueType> ins_record;
  };  // class BWDeltaInsertNode

  /*
   * class BWDeltaDeleteNode - Key delete delta
   */
  class BWDeltaDeleteNode : public BWDeltaNode {
   public:
    BWDeltaDeleteNode(BWNode* _child_node,
                      std::pair<KeyType, ValueType> _del_record)
        : BWDeltaNode(PageType::deltaDelete, _child_node) {
      del_record = _del_record;
    }

    std::pair<KeyType, ValueType> del_record;
  };  // class BWDeltaDeleteNode

  /*
   * class BWDeltaSplitNode - Leaf and inner split node
   */
  class BWDeltaSplitNode : public BWDeltaNode {
   public:
    BWDeltaSplitNode(BWNode* _child_node, KeyType _separator_key,
                     PID _split_sibling, KeyType _next_separator_key)
        : BWDeltaNode(PageType::deltaSplit, _child_node),
          separator_key(_separator_key),
          next_separator_key(_next_separator_key),
          split_sibling(_split_sibling) {}

    KeyType separator_key;
    PID split_sibling;
    KeyType next_separator_key;
  };  // class BWDeltaSplitNode

  /*
   * class BWDeltaIndexTermInsertNode - Index separator add
   */
  class BWDeltaIndexTermInsertNode : public BWDeltaNode {
   public:
    BWDeltaIndexTermInsertNode(BWNode* _child_node,
                               KeyType new_split_separator_key,
                               PID new_split_sibling,
                               KeyType next_separator_key)
        : BWDeltaNode(PageType::deltaIndexTermInsert, _child_node),
          new_split_separator_key(new_split_separator_key),
          new_split_sibling(new_split_sibling),
          next_separator_key(next_separator_key) {}

    KeyType new_split_separator_key;
    PID new_split_sibling;
    KeyType next_separator_key;
  };  // class BWDeltaIndexTermInsertNode

  /*
   * BWDeltaIndexTermDeleteNode - Remove separator in inner page
   */
  class BWDeltaIndexTermDeleteNode : public BWDeltaNode {
   public:
    BWDeltaIndexTermDeleteNode(BWNode* _child_node, PID node_to_merge_into,
                               PID node_to_remove, KeyType merge_node_low_key,
                               KeyType next_separator_key)
        : BWDeltaNode(PageType::deltaIndexTermDelete, _child_node),
          node_to_merge_into(node_to_merge_into),
          node_to_remove(node_to_remove),
          merge_node_low_key(merge_node_low_key),
          next_separator_key(next_separator_key) {}

    PID node_to_merge_into;
    PID node_to_remove;
    KeyType merge_node_low_key;
    KeyType remove_node_low_key;
    KeyType next_separator_key;
  };  // class BWDeltaIndexTermDeleteNode

  /*
   * class BWDeltaRemoveNode - Delete and free page
   *
   * NOTE: This is not delete key node
   */
  class BWDeltaRemoveNode : public BWDeltaNode {
   public:
    BWDeltaRemoveNode(BWNode* _child_node)
        : BWDeltaNode(PageType::deltaRemove, _child_node) {}
  };  // class BWDeltaRemoveNode

  /*
   * class BWDeltaMergeNode - Merge two pages into one
   */
  class BWDeltaMergeNode : public BWDeltaNode {
   public:
    BWDeltaMergeNode(BWNode* _child_node, KeyType separator_key,
                     BWNode* merge_node)
        : BWDeltaNode(PageType::deltaMerge, _child_node),
          separator_key(separator_key),
          merge_node(merge_node) {}

    PID node_to_remove;
    KeyType separator_key;
    BWNode* merge_node;
  };  // class BWDeltaMergeNode

  /*
   * class BWInnerNode - Inner node that contains separator
   *
   * Contains guide post keys for pointing to the right PID when search
   * for a key in the index
   *
   * Elastic container to allow for separation of consolidation, splitting
   * and merging
   */
  class BWInnerNode : public BWNode {
   public:
    BWInnerNode(KeyType _lower_bound)
        : BWNode(PageType::inner), lower_bound(_lower_bound) {}

    const KeyType lower_bound;
    std::vector<std::pair<KeyType, PID>> separators;
  };  // class BWInnerNode

  /*
   * class BWLeafNode - Leaf node that actually stores data
   *
   * Lowest level nodes in the tree which contain the payload/value
   * corresponding to the keys
   */
  class BWLeafNode : public BWNode {
   public:
    BWLeafNode(KeyType _lower_bound, PID _next)
        : BWNode(PageType::leaf), lower_bound(_lower_bound), next(_next) {}

    const KeyType lower_bound;
    PID next;

    // Elastic container to allow for separation of consolidation, splitting
    // and merging
    std::vector<std::pair<KeyType, ValueType>> data;
  };  // class BWLeafNode

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

  bool collectPageItem(PID page_id,
                       KeyType &key,
                       std::vector<std::pair<KeyType, ValueType>> &output,
                       PID *next_pid,
                       KeyType *highest_key);
  bool collectAllPageItem(PID page_id,
                          std::vector<std::pair<KeyType, ValueType>> &output,
                          PID *next_pid);
  bool insert(const KeyType& key, const ValueType& value);
  bool exists(const KeyType& key);
  bool erase(const KeyType& key, const ValueType& value);


  std::vector<ValueType> find(const KeyType& key);

 private:
  /*
   * isTupleEqual() - Whether two tuples are equal
   *
   * It calls key comparator and value comparator respectively
   * We need this function to determine deletion in a duplicated key
   * environment
   */
  inline bool isTupleEqual(const std::pair<KeyType, ValueType> &a,
                          const std::pair<KeyType, ValueType> &b) const {
    return key_equal(a.first, b.first) && \
           m_val_equal(a.second, b.second);
  }

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
      BWNode* node, std::vector<BWNode*>& garbage_nodes,
      std::vector<std::pair<KeyType, ValueType>>& data, PID& sibling,
      bool& has_merge, BWNode*& merge_node, KeyType& lower_bound);

  bool consolidateLeafNode(PID id);

  void traverseAndConsolidateInner(
      BWNode* node, std::vector<BWNode*>& garbage_nodes,
      std::vector<std::pair<KeyType, PID>>& separators, bool& has_merge,
      BWNode*& merge_node, KeyType& lower_bound);

  bool consolidateInnerNode(PID id);

  bool isSMO(BWNode* n);

  bool isBasePage(BWNode* node_p) { return node_p->type == leaf; }

  PID findLeafPage(const KeyType& key);

  bool splitInnerNode(PID id);

  bool splitLeafNode(PID id);

  bool mergeInnerNode(PID id);

  bool mergeLeafNode(PID id);

  // Atomically install a page into mapping table
  // NOTE: There are times that new pages are not installed into
  // mapping table but instead they directly replace other page
  // with the same PID
  PID installPage(BWNode* new_node_p);

  // This only applies to leaf node - For intermediate nodes
  // the insertion of sep/child pair must be done using different
  // insertion method
  InstallDeltaResult installDeltaInsert(PID leaf_pid, const KeyType& key,
                                        const ValueType& value);

  InstallDeltaResult installDeltaDelete(PID leaf_pid, const KeyType& key,
                                        const ValueType& value);

  InstallDeltaResult installIndexTermDeltaInsert(
      PID pid, const BWDeltaSplitNode* split_node);

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
  std::vector<std::atomic<BWNode*>> mapping_table{max_table_size};

  PID m_root;
  const KeyComparator& m_key_less;
  const ValueComparator m_val_equal;

  // Leftmost leaf page
  // NOTE: We assume the leftmost lead page will always be there
  // For split it remains to be the leftmost page
  // For merge and remove we need to make sure the last remianing page
  // shall not be removed
  // Using this pointer we could do sequential search more efficiently
  PID first_leaf;
};

/*
 * struct LessFn - Comparator for <key, value> tuples
 *
 * This is required by std::sort in order to consolidate pages
 * This function object compares tuples by its key (less than relation)
 */
template <typename KeyType, typename ValueType, class KeyComparator>
struct LessFn {
  LessFn(const KeyComparator& comp) : m_key_less(comp) {}

  bool operator()(const std::pair<KeyType, ValueType>& l,
                  const std::pair<KeyType, ValueType>& r) const {
    return m_key_less(std::get<0>(l), std::get<0>(r));
  }

  const KeyComparator& m_key_less;
};

}  // End index namespace
}  // End peloton namespace

/// //////////////////////////////////////////////////////////
/// Implementations
/// //////////////////////////////////////////////////////////

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
    : current_mapping_table_size(0), next_pid(0), m_key_less(_m_key_less),
      m_val_equal(ValueComparator()) {
  // Initialize an empty tree
  BWLeafNode* initial_leaf = new BWLeafNode(KeyType(), NONE_PID);

  PID leaf_pid = installPage(initial_leaf);
  BWInnerNode* initial_inner = new BWInnerNode(KeyType());

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
 * isSMO() - Returns true if the target is a SMO operation
 *
 * We maintain the invariant that if SMO is going to appear in delta chian,
 * then it must be the first on it. Every routine that sees an SMO on delta
 * must then consolidate it in order to append new delta record
 *
 * SMOs that could appear in any (leaf & inner) delta chain:
 *   - deltaSplit
 *   - deltaRemove
 *   - deltaMerge
 */
template <typename KeyType, typename ValueType, typename KeyComparator>
bool BWTree<KeyType, ValueType, KeyComparator>::isSMO(BWNode* n) {
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

  BWNode* leaf_node_p = mapping_table[page_pid].load();
  assert(leaf_node_p != nullptr);

  BWDeltaInsertNode* insert_page_p = nullptr;
  BWDeltaDeleteNode* delete_page_p = nullptr;
  BWLeafNode* base_page_p = nullptr;
  std::pair<KeyType, ValueType>* pair_p = nullptr;

  while (1) {
    if (leaf_node_p->type == deltaInsert) {
      insert_page_p = static_cast<BWDeltaInsertNode*>(leaf_node_p);

      bwt_printf("See DeltaInsert Page: %d\n", insert_page_p->ins_record.first);

      // If we see an insert node first, then this implies that the
      // key does exist in the future consolidated version of the page
      if (key_equal(insert_page_p->ins_record.first, key) == true) {
        return true;
      } else {
        leaf_node_p = (static_cast<BWDeltaNode*>(leaf_node_p))->child_node;
      }
    } else if (leaf_node_p->type == deltaDelete) {
      delete_page_p = static_cast<BWDeltaDeleteNode*>(leaf_node_p);

      // For delete record it implies the node has been removed
      // Ravi: This is wrong in the presence of duplicates you cannot bail out
      // on seeing one delete in the key
      if (key_equal(delete_page_p->del_record.first, key) == true) {
        return false;
      } else {
        leaf_node_p = (static_cast<BWDeltaNode*>(leaf_node_p))->child_node;
      }
    } else if (isBasePage(leaf_node_p)) {
      // The last step is to search the key in the leaf, and we search
      // for the key in leaf page
      base_page_p = static_cast<BWLeafNode*>(leaf_node_p);
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
 * collectPageItem() - Collect items in a page with a given key
 *
 * This method also returns highest key and next page PID in order
 * to facilitate page scan.
 *
 * If SMO is found in delta chain then return false, and no argument
 * will be modified. If everything goes well return true.
 */
template <typename KeyType, typename ValueType, typename KeyComparator>
bool BWTree<KeyType, ValueType, KeyComparator>::collectPageItem(
                       PID page_id,
                       KeyType &key,
                       std::vector<std::pair<KeyType, ValueType>> &output,
                       PID *next_pid,
                       KeyType *highest_key) {

  std::vector<std::pair<KeyType, ValueType>> all_data;
  bool ret = collectAllPageItem(page_id, all_data, next_pid);
  if(ret == false) {
    return false;
  }

  // Get highest key for deciding whether to search next page
  *highest_key = all_data.back().first;

  // Filter only those with the same key as specified
  for(auto it = all_data.begin();
      it != all_data.end();
      it++) {
    if(key_equal(key, (*it).first)) {
      output.push_back(*it);
    }
  }

  return true;
}

/*
 * collectPageItem() - Collect items on a given logical page (PID) without
 *                     a given key (i.e. collect all)
 *
 * It returns the PID for next page in chained pages. If we are looking
 * multiple pages this would be helpful.
 *
 */
template <typename KeyType, typename ValueType, typename KeyComparator>
bool BWTree<KeyType, ValueType, KeyComparator>::collectAllPageItem(
                                        PID page_id,
                                        std::vector<std::pair<KeyType, ValueType>> &output,
                                        PID *next_pid) {
  BWNode *node_p = mapping_table[page_id].load();
  // If we have seen a SMO then just return false
  /// NOTE: This could be removed since we guarantee the side pointer
  /// always points to the next page no matter what kind of SMO
  /// happens on top of that
  if(isSMO(node_p)) {
    return false;
  }

  int counter = 0;
  // We need to replay the delta chain
  std::stack<BWDeltaNode *> delta_stack;
  while(1) {
    // Safety measure
    if(counter++ > ITER_MAX) {
      assert(false);
    }

    // If a SMO appears then it must be the topmost element
    // in that case it would be detected on entry of the function
    assert(node_p->type == PageType::deltaInsert || \
           node_p->type == PageType::deltaDelete || \
           node_p->type == PageType::leaf);

    if(node_p->type == PageType::deltaInsert || \
       node_p->type == PageType::deltaDelete) {
          delta_stack.push(static_cast<BWDeltaNode *>(node_p));
          // Go to its child
          node_p = (static_cast<BWDeltaNode *>(node_p))->child_node;
    } else if(node_p->type == PageType::leaf) {
      break;
    }
  } /// while(1)

  // When we reach here we know there is a leaf node
  BWLeafNode *leaf_node_p = static_cast<BWLeafNode *>(node_p);
  /// Set output variable to enable quick search
  *next_pid = leaf_node_p->next;

  // Bulk load
  std::vector<std::pair<KeyType, ValueType>> linear_data(leaf_node_p->data.begin(), leaf_node_p->data.end());
  // boolean vector to decide whether a pair has been deleted or not
  std::vector<bool> deleted_flag(leaf_node_p->data.size(), false);

  bool ever_deleted = false;
  // Replay delta chain
  while(delta_stack.size() > 0) {
    BWDeltaNode *delta_node_p = delta_stack.top();
    delta_stack.pop();

    node_p = static_cast<BWNode *>(delta_node_p);

    if(node_p->type == PageType::deltaInsert) {
      BWDeltaInsertNode *delta_insert_node_p = \
        static_cast<BWDeltaInsertNode *>(node_p);

      linear_data.push_back(delta_insert_node_p->ins_record);
      deleted_flag.push_back(false);
    } else if(node_p->type == PageType::deltaDelete) {
      BWDeltaDeleteNode *delta_delete_node_p = \
        static_cast<BWDeltaDeleteNode *>(node_p);

      assert(deleted_flag.size() == linear_data.size());
      int len = deleted_flag.size();
      for(int i = 0;i < len;i++) {
        // If some entry metches deleted record, then we think it is deleted
        if(isTupleEqual(linear_data[i], delta_delete_node_p->del_record)) {
          deleted_flag[i] = true;
          ever_deleted = true;
        }
      }
    }
  }

  // Less than relation function object for sorting
  using LessFnT = LessFn<KeyType, ValueType, KeyComparator>;
  LessFnT less_fn(m_key_less);

  // If there is no deletion then we know the data is OK, it just needs to be sorted
  if(ever_deleted == false) {
    std::sort(linear_data.begin(), linear_data.end(), less_fn);
    output.insert(output.begin(), linear_data.begin(), linear_data.end());
  } else {
    assert(deleted_flag.size() == linear_data.size());
    int len = deleted_flag.size();

    // Otherwise we have to insert them into output buffer one by one
    for(int i = 0;i < len;i++) {
      if(deleted_flag[i] == false) {
        output.push_back(linear_data[i]);
      }
    }

    std::sort(output.begin(), output.end(), less_fn);
  }

  return true;
}

/*
 * insert() - Insert a key-value pair into B-Tree
 *
 * NOTE: No duplicated key support
 */
template <typename KeyType, typename ValueType, typename KeyComparator>
bool BWTree<KeyType, ValueType, KeyComparator>::insert(const KeyType& key,
                                                       const ValueType& value) {
  bool insert_success = false;
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
  bool delete_success = false;
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
  BWNode* curr_node = mapping_table[curr_pid].load();

  PID parent_pid = this->NONE_PID;
  int chain_length = 0; // Length of delta chain, including current node

  // Trigger consolidation

  // Trigger structure modifying operations
  // split, remove, merge

  while (1) {
    assert(curr_node != nullptr);
    chain_length += 1;
    ////////////////////////////////////////////////////////////////////////////
    /// Inner
    if (curr_node->type == PageType::inner) {
      bwt_printf("Traversing inner node\n");

      BWInnerNode* inner_node = static_cast<BWInnerNode*>(curr_node);
      // The consolidate has to ensure that it does not leave empty
      // inner nodes
      assert(inner_node->separators.size() > 0);

      if (chain_length > DELTA_CHAIN_INNER_THRESHOLD) {
        consolidateInnerNode(curr_pid);
        // Reset the chain length counter
        chain_length = 0;
        // Even if the consolidate fails or completes the search needs to
        // fetch the curr_node from the mapping table
        curr_node = mapping_table[curr_pid].load();
        continue;
      }

      if (inner_node->separators.size() < INNER_NODE_SIZE_MIN) {
        // Install a remove delta on top of the node
        installDeltaRemove(curr_pid);
        curr_node = mapping_table[curr_pid].load();
        continue;
      }

      if (inner_node->separators.size() > this->INNER_NODE_SIZE_MAX) {
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
          chain_length = 0;
          // chain_length should be zero when you reach an inner node
          assert(chain_length == 0);
          curr_pid = inner_node->separators[i - 1].second;
          break;
        }
      }

      if (!found_sep) {
        // Change page
        parent_pid = curr_pid;
        chain_length = 0;
        // This should be zero when you reach an inner node
        assert(chain_length == 0);
        curr_pid = inner_node->separators.back().second;
      }

      curr_node = mapping_table[curr_pid].load();
      continue;

    ////////////////////////////////////////////////////////////////////////////
    /// Index Term Insert
    } else if (curr_node->type == PageType::deltaIndexTermInsert) {
      bwt_printf("Traversing index term insert node\n");
      BWDeltaIndexTermInsertNode* index_insert_delta =
          static_cast<BWDeltaIndexTermInsertNode*>(curr_node);
      // Increment the index delta chain counter
      chain_length++;
      if (chain_length > this->DELTA_CHAIN_INNER_THRESHOLD) {
        consolidateInnerNode(curr_pid);
        // Reset the chain length counter
        chain_length = 0;
        // Even if the consolidate fails or completes the search needs to
        // fetch the curr_node from the mapping table
        curr_node = mapping_table[curr_pid].load();
        continue;
      }
      if (key_greater(key, index_insert_delta->new_split_separator_key) &&
          key_lessequal(key, index_insert_delta->next_separator_key)) {
        // Change page
        parent_pid = curr_pid;
        chain_length = 0;
        // chain_length should be zero when you reach an inner node
        curr_pid = index_insert_delta->new_split_sibling;
        curr_node = mapping_table[curr_pid].load();
        continue;
      }

      curr_node = index_insert_delta->child_node;

    ////////////////////////////////////////////////////////////////////////////
    /// Index Term Delete
    } else if (curr_node->type == PageType::deltaIndexTermDelete) {
      bwt_printf("Traversing index term delete node\n");
      BWDeltaIndexTermDeleteNode* index_delete_delta =
          static_cast<BWDeltaIndexTermDeleteNode*>(curr_node);
      // Increment the index delta chain counter
      chain_length++;
      if (chain_length > this->DELTA_CHAIN_INNER_THRESHOLD) {
        consolidateInnerNode(curr_pid);
        // Reset the chain length counter
        chain_length = 0;
        // Even if the consolidate fails or completes the search needs to
        // fetch the curr_node from the mapping table
        curr_node = mapping_table[curr_pid].load();
        continue;
      }
      if (key_greater(key, index_delete_delta->merge_node_low_key) &&
          key_lessequal(key, index_delete_delta->next_separator_key)) {
        // Change page
        parent_pid = curr_pid;
        chain_length = 0;
        // chain_length should be zero when you reach an inner node
        curr_pid = index_delete_delta->node_to_merge_into;
        curr_node = mapping_table[curr_pid].load();
        continue;
      }

      curr_node = index_delete_delta->child_node;

    ////////////////////////////////////////////////////////////////////////////
    /// Split
    } else if (curr_node->type == PageType::deltaSplit) {
      bwt_printf("Traversing split node\n");
      // Our invariant is that there should be no delta chains on top of a
      // split node
      assert(chain_length == 0 && chain_length == 0);
      BWDeltaSplitNode* split_delta = static_cast<BWDeltaSplitNode*>(curr_node);

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

    ////////////////////////////////////////////////////////////////////////////
    /// Remove
    } else if (curr_node->type == PageType::deltaRemove) {
      bwt_printf("Traversing remove node\n");
      // Note: should not trigger a remove on the left most leaf even if the
      // number of tuples is below a threshold

      // TODO: Install an installDeltaMerge on sibling and retry till it
      // succeds. Thread
      // cannot proceed until this succeeds
      assert(false);
    ////////////////////////////////////////////////////////////////////////////
    /// Merge
    } else if (curr_node->type == PageType::deltaMerge) {
      bwt_printf("Traversing merge node\n");
      // Our invariant is that there should be no delta chains on top of a
      // merge node
      assert(chain_length == 0);
      BWDeltaMergeNode* merge_delta = static_cast<BWDeltaMergeNode*>(curr_node);

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
    ////////////////////////////////////////////////////////////////////////////
    /// Leaf Insert
    } else if (curr_node->type == PageType::deltaInsert) {
      bwt_printf("Traversing insert node\n");
      BWDeltaInsertNode* insert_node =
          static_cast<BWDeltaInsertNode*>(curr_node);
      if (key_lessequal(key, insert_node->ins_record.first)) {
        break;
      } else {
        curr_node = insert_node->child_node;
        assert(curr_node != nullptr);
      }
    ////////////////////////////////////////////////////////////////////////////
    /// Leaf Delete
    } else if (curr_node->type == PageType::deltaDelete) {
      bwt_printf("Traversing delete node\n");
      BWDeltaDeleteNode* delete_node =
          static_cast<BWDeltaDeleteNode*>(curr_node);
      if (key_lessequal(key, delete_node->del_record.first)) {
        break;
      } else {
        curr_node = delete_node->child_node;
        assert(curr_node != nullptr);
      }
    ////////////////////////////////////////////////////////////////////////////
    /// Leaf
    } else if (curr_node->type == PageType::leaf) {
      bwt_printf("Traversing leaf node\n");

      BWLeafNode* leaf_node = static_cast<BWLeafNode*>(curr_node);

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

  BWNode* curr_node = mapping_table[leaf_page].load();

  // Check if the node is marked for consolidation, splitting or merging
  // BWNode* next_node = nullptr;
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
void BWTree<KeyType, ValueType, KeyComparator>::traverseAndConsolidateLeaf(
    BWNode* original_node, std::vector<BWNode*>& garbage_nodes,
    std::vector<std::pair<KeyType, ValueType>>& data, PID& sibling,
    bool& has_merge, BWNode*& merge_node, KeyType& lower_bound) {
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

  BWNode* node = original_node;
  while (node->type != leaf) {
    switch (node->type) {
      case deltaInsert: {
        BWDeltaInsertNode* insert_node = static_cast<BWDeltaInsertNode*>(node);
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
        BWDeltaDeleteNode* delete_node = static_cast<BWDeltaDeleteNode*>(node);
        delete_records.insert(delete_node->del_record);
        break;
      }
      case deltaSplit: {
        // Our invariant is that split nodes always force a consolidate, so
        // should be at the top
        assert(node == original_node);  // Ensure the split is at the top
        assert(!has_split);             // Ensure this is the only split
        BWDeltaSplitNode* split_node = static_cast<BWDeltaSplitNode*>(node);
        has_split = true;
        split_separator_key = split_node->separator_key;
        new_sibling = split_node->split_sibling;
        break;
      }
      case deltaMerge: {
        // Same as split, invariant is that merge nodes always force a
        // consolidate, so should be at the top
        assert(node == original_node);
        BWDeltaMergeNode* merge_delta = static_cast<BWDeltaMergeNode*>(node);
        has_merge = true;
        merge_separator_key = merge_delta->separator_key;
        merge_node = merge_delta->merge_node;
        break;
      }
      default:
        assert(false);
    }

    garbage_nodes.push_back(node);
    node = static_cast<BWDeltaNode*>(node)->child_node;
    assert(node != nullptr);
  }

  // node is a leaf node
  BWLeafNode* leaf_node = static_cast<BWLeafNode*>(node);

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
  BWNode* original_node = mapping_table[id].load();

  std::vector<BWNode*> garbage_nodes;
  std::vector<std::pair<KeyType, ValueType>> data;
  PID sibling;
  bool has_merge = false;
  BWNode* merge_node = nullptr;
  KeyType lower_bound;
  traverseAndConsolidateLeaf(original_node, garbage_nodes, data, sibling,
                             has_merge, merge_node, lower_bound);
  if (has_merge) {
    BWNode* dummy_node;
    KeyType dummy_bound;
    traverseAndConsolidateLeaf(merge_node, garbage_nodes, data, sibling,
                               has_merge, dummy_node, dummy_bound);
    assert(!has_merge);
  }

  BWLeafNode* consolidated_node = new BWLeafNode(lower_bound, sibling);
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
    BWNode* original_node, std::vector<BWNode*>& garbage_nodes,
    std::vector<std::pair<KeyType, PID>>& separators, bool& has_merge,
    BWNode*& merge_node, KeyType& lower_bound) {
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
  BWNode* node = original_node;
  while (node->type != inner) {
    switch (node->type) {
      case deltaIndexTermInsert: {
        BWDeltaIndexTermInsertNode* insert_node =
            static_cast<BWDeltaIndexTermInsertNode*>(node);
        if (!has_split || key_less(insert_node->new_split_separator_key,
                                   split_separator_key)) {
          insert_separators.push_back({insert_node->new_split_separator_key,
                                       insert_node->new_split_sibling});
        }
        break;
      }
      case deltaIndexTermDelete: {
        BWDeltaIndexTermDeleteNode* delete_node =
            static_cast<BWDeltaIndexTermDeleteNode*>(node);
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
        BWDeltaSplitNode* split_node = static_cast<BWDeltaSplitNode*>(node);
        has_split = true;
        split_separator_key = split_node->separator_key;
        break;
      }
      case deltaMerge: {
        // Same as split, invariant is that merge nodes always force a
        // consolidate, so should be at the top
        assert(node == original_node);
        BWDeltaMergeNode* merge_delta = static_cast<BWDeltaMergeNode*>(node);
        has_merge = true;
        merge_separator_key = merge_delta->separator_key;
        merge_node = merge_delta->merge_node;
        break;
      }
      default:
        assert(false);
    }

    garbage_nodes.push_back(node);
    node = static_cast<BWDeltaNode*>(node)->child_node;
  }

  BWInnerNode* inner_node = static_cast<BWInnerNode*>(node);

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
  BWNode* original_node = mapping_table[id].load();

  // Keep track of nodes so we can garbage collect later
  std::vector<BWNode*> garbage_nodes;
  std::vector<std::pair<KeyType, PID>> separators;
  bool has_merge = false;
  BWNode* merge_node = nullptr;
  KeyType lower_bound;
  traverseAndConsolidateInner(original_node, garbage_nodes, separators,
                              has_merge, merge_node, lower_bound);
  if (has_merge) {
    BWNode* dummy_node;
    KeyType dummy_bound;
    traverseAndConsolidateInner(merge_node, garbage_nodes, separators,
                                has_merge, dummy_node, dummy_bound);
    assert(!has_merge);
  }

  BWInnerNode* consolidated_node = new BWInnerNode(lower_bound);
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
// NOTE: This implementation referred to the BW-Tree implementation on github:
// >> https://github.com/flode/BWTree/blob/master/bwtree.hpp
template <typename KeyType, typename ValueType, class KeyComparator>
typename BWTree<KeyType, ValueType, KeyComparator>::PID
BWTree<KeyType, ValueType, KeyComparator>::installPage(BWNode* new_node_p) {
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

  BWNode* old_leaf_p = mapping_table[leaf_pid].load();

  if (isSMO(old_leaf_p)) {
    return install_need_consolidate;
  }

  BWNode* new_leaf_p = (BWNode*)new BWDeltaInsertNode(old_leaf_p, ins_record);

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

  BWNode* old_leaf_p = mapping_table[leaf_pid].load();

  if (isSMO(old_leaf_p)) {
    return install_need_consolidate;
  }

  BWNode* new_leaf_p =
      (BWNode*)new BWDeltaDeleteNode(old_leaf_p, delete_record);

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
    __attribute__((unused)) const BWDeltaSplitNode* split_node) {
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

