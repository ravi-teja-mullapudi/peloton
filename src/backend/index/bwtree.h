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

template <typename KeyType, typename ValueType, typename KeyComparator>
struct LessFnT;

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
  constexpr static unsigned int DELTA_CHAIN_LEAF_THRESHOLD = 8;
  // Node sizes for triggering splits and merges on inner nodes
  constexpr static unsigned int INNER_NODE_SIZE_MIN = 0;
  constexpr static unsigned int INNER_NODE_SIZE_MAX = 8;
  // Node sizes for triggering splits and merges on leaf nodes
  constexpr static unsigned int LEAF_NODE_SIZE_MIN = 0;
  constexpr static unsigned int LEAF_NODE_SIZE_MAX = 31;
  // Debug constant: The maximum number of iterations we could do
  // It prevents dead loop hopefully
  constexpr static int ITER_MAX = 99999;

  // Enumeration of the types of nodes required in updating both the values
  // and the index in the BW Tree. Currently only adding node types for
  // supporting splits.

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
    virtual ~BWNode() {}

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
    KeyType next_separator_key;
    PID split_sibling;
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
    BWInnerNode(KeyType _lower_bound, KeyType _upper_bound)
        : BWNode(PageType::inner),
          lower_bound(_lower_bound),
          upper_bound(_upper_bound) {}

    const KeyType lower_bound;
    const KeyType upper_bound;
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
    BWLeafNode(KeyType _lower_bound, KeyType _upper_bound, PID _next)
        : BWNode(PageType::leaf),
          lower_bound(_lower_bound),
          upper_bound(_upper_bound),
          next(_next) {}

    const KeyType lower_bound;
    const KeyType upper_bound;
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
    bool operator()(const ItemPointer& a, const ItemPointer& b) const {
      return (a.block == b.block) && (a.offset == b.offset);
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
  // Scan all values
  void getAllValues(std::vector<ValueType>& result);

 private:
  bool collectPageItem(BWNode* node_p, const KeyType& key,
                       std::vector<std::pair<KeyType, ValueType>>& output,
                       PID* next_pid, KeyType* highest_key);
  bool collectAllPageItem(BWNode* node_p,
                          std::vector<std::pair<KeyType, ValueType>>& output,
                          PID* next_pid);

  /*
   * isValueEqual() - Compare two values and see if they are equal
   */
  inline bool isValueEqual(const ValueType &a, const ValueType &b) const {
    return m_val_equal(a, b);
  }

  /*
   * isTupleEqual() - Whether two tuples are equal
   *
   * It calls key comparator and value comparator respectively
   * We need this function to determine deletion in a duplicated key
   * environment
   */
  inline bool isTupleEqual(const std::pair<KeyType, ValueType>& a,
                           const std::pair<KeyType, ValueType>& b) const {
    return key_equal(a.first, b.first) && m_val_equal(a.second, b.second);
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

  // Input invariants:
  // 1. All insert records should end up in the final output. That means that
  //    any intersection with the delete records has already been done
  // 2. The insert records should be sorted on key type
  // 3. Data is sorted on key tyep
  template <typename Key, typename Value, typename Compare>
  void consolidateModifications(
      std::vector<std::pair<Key, Value>>& insert_records,
      std::set<std::pair<Key, Value>, Compare>& delete_records,
      typename std::vector<std::pair<Key, Value>>::iterator data_start,
      typename std::vector<std::pair<Key, Value>>::iterator data_end,
      Compare& less_fn, std::vector<std::pair<Key, Value>>& output_data);

  // Internal functions to be implemented
  void traverseAndConsolidateLeaf(
      BWNode* node, std::vector<BWNode*>& garbage_nodes,
      std::vector<std::pair<KeyType, ValueType>>& data, PID& sibling,
      bool& has_merge, BWNode*& merge_node, KeyType& lower_bound,
      KeyType& upper_bound);

  bool consolidateLeafNode(PID id, BWNode* pid_node);

  void traverseAndConsolidateInner(
      BWNode* node, std::vector<BWNode*>& garbage_nodes,
      std::vector<std::pair<KeyType, PID>>& separators, bool& has_merge,
      BWNode*& merge_node, KeyType& lower_bound, KeyType& upper_bound);

  bool consolidateInnerNode(PID id, BWNode* pid_node);

  bool isLeaf(BWNode* node);

  bool performConsolidation(PID id, BWNode* node);

  bool isSMO(BWNode* n);

  std::pair<PID, BWNode*> findLeafPage(const KeyType& key);

  bool splitInnerNode(PID id);

  bool splitLeafNode(PID id);

  bool mergeInnerNode(PID id);

  bool mergeLeafNode(PID id);

  BWNode* spinOnSMOByKey(KeyType& key);
  BWNode* spinOnSMOByPID(PID page_id);

  // Atomically install a page into mapping table
  // NOTE: There are times that new pages are not installed into
  // mapping table but instead they directly replace other page
  // with the same PID
  PID installPage(BWNode* new_node_p);

  // This only applies to leaf node - For intermediate nodes
  // the insertion of sep/child pair must be done using different
  // insertion method
  InstallDeltaResult installDeltaInsert(PID leaf_pid, BWNode* leaf_ndoe,
                                        const KeyType& key,
                                        const ValueType& value);

  InstallDeltaResult installDeltaDelete(PID leaf_pid, BWNode* leaf_node,
                                        const KeyType& key,
                                        const ValueType& value);

  InstallDeltaResult installIndexTermDeltaInsert(
      PID pid, BWNode* inner_node, const BWDeltaSplitNode* split_node);

  InstallDeltaResult installIndexTermDeltaDelete(PID pid, BWNode* inner_node,
                                                 const KeyType& key,
                                                 const ValueType& value);

  // Functions to install SMO deltas
  InstallDeltaResult installDeltaMerge(PID node, PID sibling);

  void deleteDeltaChain(BWNode* node);

  void addGarbageNodes(std::vector<BWNode*>& garbage);

  /// //////////////////////////////////////////////////////////////
  /// Data member definition
  /// //////////////////////////////////////////////////////////////

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

  // Not efficient but just for correctness
  std::mutex garbage_mutex;
  std::vector<BWNode*> garbage_nodes;

  std::atomic<PID> m_root;
  const KeyComparator m_key_less;
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
    : current_mapping_table_size(0),
      next_pid(0),
      m_key_less(_m_key_less),
      m_val_equal(ValueComparator()) {
  // Initialize an empty tree
  KeyType low_key = KeyType::NEG_INF_KEY;
  KeyType high_key = KeyType::POS_INF_KEY;

  BWLeafNode* initial_leaf = new BWLeafNode(low_key, high_key, NONE_PID);
  PID leaf_pid = installPage(initial_leaf);

  BWInnerNode* initial_inner = new BWInnerNode(low_key, high_key);
  initial_inner->separators.emplace_back(low_key, leaf_pid);
  PID inner_pid = installPage(initial_inner);

  m_root = inner_pid;
  first_leaf = leaf_pid;

  bwt_printf("Init: Initializer returns. Leaf = %lu, inner = %lu\n", leaf_pid,
             inner_pid);
}

/*
 * Destructor - Free up all pages
 */
template <typename KeyType, typename ValueType, typename KeyComparator>
BWTree<KeyType, ValueType, KeyComparator>::~BWTree() {
  // TODO: finish this
  for (std::atomic<BWNode*>& atm_node : mapping_table) {
    BWNode* node = atm_node.load();
    // If this is a remove node, we need to check if the merge was installed
    // because otherwise we won't cleanup the data under the remove node
    deleteDeltaChain(node);
  }
  for (BWNode* node : garbage_nodes) {
    delete node;
  }
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
 * spinOnSMOByPID() - Keeps checking whether a PID is SMO, and call
 *                    consolidation if it is
 *
 * This is used to make sure a logical page pointer always points to
 * a sequential structure
 */
template <typename KeyType, typename ValueType, typename KeyComparator>
typename BWTree<KeyType, ValueType, KeyComparator>::BWNode*
BWTree<KeyType, ValueType, KeyComparator>::spinOnSMOByPID(PID page_id) {
  int counter = 0;
  BWNode* node_p = nullptr;

  node_p = mapping_table[page_id];
  assert(node_p != nullptr);

  while (isSMO(node_p)) {
    if (counter++ > ITER_MAX) {
      assert(false);
    }

    bool ret = consolidateLeafNode(page_id, node_p);
    if (ret == false) {
      /// Nothing to do?
    }

    node_p = mapping_table[page_id];
  }

  // The returned page is guaranteed not to be SMO
  /// Even if some other operation adds SMO on top of that
  /// we could only see the physical pointer
  return node_p;
}

/*
 * spinOnSMOByKey() - This method keeps finding page if it sees a SMO on
 * the top of a leaf delta chain
 *
 * Since findLeafPage() will do all of the consolidation work, we just
 * need to reinvoke routine
 */
template <typename KeyType, typename ValueType, typename KeyComparator>
typename BWTree<KeyType, ValueType, KeyComparator>::BWNode*
BWTree<KeyType, ValueType, KeyComparator>::spinOnSMOByKey(KeyType& key) {
  int counter = 0;
  BWNode* node_p = nullptr;

  // Find the first page where the key lies in
  PID page_id;
  std::tie(page_id, node_p) = findLeafPage(key);
  (void)page_id;

  // The returned page is guaranteed not to be SMO
  /// Even if some other operation adds SMO on top of that
  /// we could only see the physical pointer
  return node_p;
}

/*
 * getAllValues() - Get all values in the tree, using key order
 *
 * This function could not use findLeafPage() since there is no key
 * relationship to use. It has to conduct its own process of traversing
 * through SMOs
 *
 * We actually do not do anything special regarding SMOs except split
 * where we could not follow the sibling pointer to get to next sibling page
 * and in that case it is this function's duty to track two branches
 * of the split and merge them into one array
 */
template <typename KeyType, typename ValueType, typename KeyComparator>
void BWTree<KeyType, ValueType, KeyComparator>::getAllValues(
    std::vector<ValueType>& result) {
  bwt_printf("Get All Values!\n");

  BWNode* node_p = nullptr;
  PID next_pid = first_leaf;

  int counter = 0;
  while (1) {
    if (counter++ > ITER_MAX) {
      assert(false);
    }

    if(next_pid == NONE_PID) {
      bwt_printf("End of leaf chain, return!\n");

      break;
    }

    std::vector<std::pair<KeyType, ValueType>> output;

    node_p = mapping_table[next_pid].load();
    if (node_p->type == PageType::deltaRemove) {
      node_p = (static_cast<BWDeltaNode*>(node_p))->child_node;
    } else if(node_p->type == PageType::deltaMerge) {
      // For deltaMerge node just let it be, the right sibling pointer
      // would find the correct leaf page
      node_p = (static_cast<BWDeltaNode*>(node_p))->child_node;
    } else if(node_p->type == PageType::deltaSplit) {
      BWDeltaSplitNode *split_node_p = static_cast<BWDeltaSplitNode*>(node_p);

      /// next_pid is not used here since we already know. Also no-SMO under split
      bool ret1 = collectAllPageItem(split_node_p->child_node, output, &next_pid);
      (void)ret1;
      assert(ret1 == true);

      PID split_sibling_pid = split_node_p->split_sibling;
      BWNode *split_sibling_p = mapping_table[split_sibling_pid];

      std::vector<std::pair<KeyType, ValueType>> output_2;
      PID next_pid_2{0};
      /// There will not be other SMOs under split(), since after split record
      /// has been posted, every SMO must first see the split() and then fail
      /// So we could safely assume the delta chain under split is linear and non-SMO
      /// Even no remove could appear under split()
      bool ret2 = collectAllPageItem(split_sibling_p, output_2, &next_pid);
      (void)ret2;
      assert(ret2 == true);

      bwt_printf("nextpid = %lu, next_pid_2 = %lu\n", next_pid, next_pid_2);
      /// Since in a consistent state these two should points to the same page (right)
      /// sib of the page before split
      //assert(next_pid == next_pid_2);


      // We do not allow empty page. If this happens then must be error
      /// NOTE: Is it possible that a page is removed too quickly so that
      /// it becomes empty before findLeafPage() could detect that?
      assert(output.size() != 0);
      assert(output_2.size() != 0);

      // Then conbine them together into one
      for(auto it = output.begin();
          it != output.end();
          it++) {
        result.push_back((*it).second);
      }

      for(auto it = output_2.begin();
          it != output_2.end();
          it++) {
        result.push_back((*it).second);
      }
    } else {
      /// After this point, node_p points to a linear delta chain
      bool ret = collectAllPageItem(node_p, output, &next_pid);
      (void)ret;
      assert(ret == true);

      assert(output.size() != 0);

      // Copy the entire array
      for (auto it = output.begin(); it != output.end(); it++) {
        result.push_back((*it).second);
      }

    } // if nodetype == ...
  } // while(1)

  return;
}

/*
 * exists() - Return true if a key exists in the tree
 *
 * Searches through the chain of delta pages, scanning for both
 * delta record and the final leaf record
 *
 * NOTE: Currently this method does support duplicated key
 * test, since we traverse multiple pages
 */
template <typename KeyType, typename ValueType, typename KeyComparator>
bool BWTree<KeyType, ValueType, KeyComparator>::exists(const KeyType& key) {
  bwt_printf("key = %d\n", key);

  /// We are guaranteed there will not be SMO in the delta chain after
  BWNode* node_p = spinOnSMOByKey(key);

  bool found = false;
  int counter = 0;
  while (1) {
    if (counter++ > ITER_MAX) {
      assert(false);
    }
    /// After this point, node_p points to a linear delta chain
    std::vector<std::pair<KeyType, ValueType>> output;

    PID next_pid;
    KeyType highest_key = KeyType();

    // This must succeed
    bool ret = collectPageItem(node_p, key, output, &next_pid, &highest_key);
    assert(ret == true);

    // If current is less than highest key, then we know the key only exists in
    // current leaf page and its delta chain
    if (key_less(key, highest_key)) {
      // In that case the output is all of possible
      found = output.size() != 0;

      break;
    } else if (key_equal(key, highest_key, key)) {
      // Trivial: highest key is the same as search key
      found = true;

      break;
    } else {
      /// If there is no next page, then we are here because the key
      /// is greater then the largest key in the tree
      if (next_pid == NONE_PID) {
        found = false;

        break;
      }

      node_p = mapping_table[next_pid].load();

      if (node_p->type == PageType::deltaRemove) {
        node_p = (static_cast<BWDeltaNode*>(node_p))->child_node;
      } else {
        /// highest key is smaller than current key, probably we missed a split
        node_p = spinOnSMOByPID(next_pid);
      }
    }  /// if(...)
  }    /// while(1)

  return found;
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
    BWNode* node_p, const KeyType& key,
    std::vector<std::pair<KeyType, ValueType>>& output, PID* next_pid,
    KeyType* highest_key) {
  std::vector<std::pair<KeyType, ValueType>> all_data;
  bool ret = collectAllPageItem(node_p, all_data, next_pid);
  if (ret == false) {
    return false;
  }

  // There should not be empty page
  assert(all_data.size() > 0);

  // Get highest key for deciding whether to search next page
  *highest_key = all_data.back().first;

  // Filter only those with the same key as specified
  for (auto it = all_data.begin(); it != all_data.end(); it++) {
    if (key_equal(key, (*it).first)) {
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
    BWNode* node_p, std::vector<std::pair<KeyType, ValueType>>& output,
    PID* next_pid) {
  /// NOTE: This could be removed since we guarantee the side pointer
  /// always points to the next page no matter what kind of SMO
  /// happens on top of that
  if (isSMO(node_p)) {
    return false;
  }

  int counter = 0;
  // We need to replay the delta chain
  std::stack<BWDeltaNode*> delta_stack;
  while (1) {
    // Safety measure
    if (counter++ > ITER_MAX) {
      assert(false);
    }

    // If a SMO appears then it must be the topmost element
    // in that case it would be detected on entry of the function
    assert(node_p->type == PageType::deltaInsert ||
           node_p->type == PageType::deltaDelete ||
           node_p->type == PageType::leaf);

    if (node_p->type == PageType::deltaInsert ||
        node_p->type == PageType::deltaDelete) {
      delta_stack.push(static_cast<BWDeltaNode*>(node_p));
      // Go to its child
      node_p = (static_cast<BWDeltaNode*>(node_p))->child_node;
    } else if (node_p->type == PageType::leaf) {
      break;
    }
  }  /// while(1)

  // When we reach here we know there is a leaf node
  BWLeafNode* leaf_node_p = static_cast<BWLeafNode*>(node_p);
  /// Set output variable to enable quick search
  *next_pid = leaf_node_p->next;

  // Bulk load
  std::vector<std::pair<KeyType, ValueType>> linear_data(
      leaf_node_p->data.begin(), leaf_node_p->data.end());

  // boolean vector to decide whether a pair has been deleted or not
  std::vector<bool> deleted_flag(leaf_node_p->data.size(), false);

  bool ever_deleted = false;
  // Replay delta chain
  while (delta_stack.size() > 0) {
    BWDeltaNode* delta_node_p = delta_stack.top();
    delta_stack.pop();

    node_p = static_cast<BWNode*>(delta_node_p);

    if (node_p->type == PageType::deltaInsert) {
      BWDeltaInsertNode* delta_insert_node_p =
          static_cast<BWDeltaInsertNode*>(node_p);

      linear_data.push_back(delta_insert_node_p->ins_record);
      deleted_flag.push_back(false);
    } else if (node_p->type == PageType::deltaDelete) {
      BWDeltaDeleteNode* delta_delete_node_p =
          static_cast<BWDeltaDeleteNode*>(node_p);

      assert(deleted_flag.size() == linear_data.size());
      int len = deleted_flag.size();
      for (int i = 0; i < len; i++) {
        // If some entry metches deleted record, then we think it is deleted
        if (isTupleEqual(linear_data[i], delta_delete_node_p->del_record)) {
          deleted_flag[i] = true;
          ever_deleted = true;
        }
      }
    }  // if
  }    // while

  // Less than relation function object for sorting
  using LessFnT = LessFn<KeyType, ValueType, KeyComparator>;
  LessFnT less_fn(m_key_less);

  // If there is no deletion then we know the data is OK, it just needs to be
  // sorted
  if (ever_deleted == false) {
    std::sort(linear_data.begin(), linear_data.end(), less_fn);
    output.insert(output.begin(), linear_data.begin(), linear_data.end());
  } else {
    assert(deleted_flag.size() == linear_data.size());
    int len = deleted_flag.size();

    // Otherwise we have to insert them into output buffer one by one
    for (int i = 0; i < len; i++) {
      if (deleted_flag[i] == false) {
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
    PID page_pid;
    BWNode* node;
    std::tie(page_pid, node) = findLeafPage(key);

    // Then install an insertion record
    InstallDeltaResult result = installDeltaInsert(page_pid, node, key, value);
    if (result == install_need_consolidate) {
      insert_success = false;
      bool consolidation_success = performConsolidation(page_pid, node);

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
    PID page_pid;
    BWNode* node;
    std::tie(page_pid, node) = findLeafPage(key);

    // Then install an insertion record
    InstallDeltaResult result = installDeltaDelete(page_pid, node, key, value);
    if (result == install_need_consolidate) {
      delete_success = false;
      performConsolidation(page_pid, node);
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

template <typename KeyType, typename ValueType, class KeyComparator>
std::vector<ValueType> BWTree<KeyType, ValueType, KeyComparator>::find(
    const KeyType& key) {
  // Hacky way to get std::set to compare values for item pointers
  auto val_cmp = [=](const ValueType& val1, const ValueType& val2) {
    return !m_val_equal(val1, val2);
  };

  std::vector<ValueType> insert_values;
  std::set<ValueType, decltype(val_cmp)> delete_values(val_cmp);

  std::vector<ValueType> values;
  // Find the leaf page the key can possibly map into

  bool retry;
  do {
    retry = false;
    PID leaf_page;
    BWNode* curr_node;
    std::tie(leaf_page, curr_node) = findLeafPage(key);

    while (curr_node != nullptr) {
      switch (curr_node->type) {
        case PageType::deltaRemove: {
          curr_node = nullptr;
          retry = true;
          break;
        }
        case PageType::deltaSplit: {
          BWDeltaSplitNode* split_node =
              static_cast<BWDeltaSplitNode*>(curr_node);
          if (key_greater(key, split_node->separator_key)) {
            // This means we need to traverse to sibling... for now, we should
            // retraverse because this probably means something jumped in
            // before us
            curr_node = nullptr;
            retry = true;
          } else {
            curr_node = split_node->child_node;
          }
          break;
        }
        case PageType::deltaMerge: {
          BWDeltaMergeNode* merge_node =
              static_cast<BWDeltaMergeNode*>(curr_node);
          if (key_greater(key, merge_node->separator_key)) {
            curr_node = merge_node->merge_node;
          } else {
            curr_node = merge_node->child_node;
          }
          break;
        }
        case PageType::deltaInsert: {
          BWDeltaInsertNode* node = static_cast<BWDeltaInsertNode*>(curr_node);

          if (key_equal(key, node->ins_record.first)) {
            auto it = delete_values.find(node->ins_record.second);
            if (it == delete_values.end()) {
              insert_values.push_back(node->ins_record.second);
            }
          }
          curr_node = node->child_node;
          break;
        }
        case PageType::deltaDelete: {
          BWDeltaDeleteNode* node = static_cast<BWDeltaDeleteNode*>(curr_node);
          if (key_equal(key, node->del_record.first)) {
            delete_values.insert(node->del_record.second);
          }
          curr_node = node->child_node;
          break;
        }
        case PageType::leaf: {
          BWLeafNode* node = static_cast<BWLeafNode*>(curr_node);

          using LessFnT = LessFn<KeyType, ValueType, KeyComparator>;
          LessFnT less_fn(m_key_less);
          auto bounds =
              std::equal_range(node->data.begin(), node->data.end(),
                               std::make_pair(key, ValueType{}), less_fn);
          for (auto it = bounds.first; it != bounds.second; ++it) {
            auto del_it = delete_values.find(it->second);
            if (del_it == delete_values.end()) {
              values.push_back(it->second);
            }
          }
          values.insert(values.end(), insert_values.begin(),
                        insert_values.end());
          curr_node = nullptr;
          break;
        }
        case PageType::deltaIndexTermInsert:
        case PageType::deltaIndexTermDelete:
        case PageType::inner: {
          // Could our PID have been replaced with an index node via a remove
          // and then subsequent split?
          bwt_printf("Find got index page for PID, retrying...\n");
          curr_node = nullptr;
          retry = true;
          break;
        }
        default:
          assert(false);
      }
    }
  } while (retry);

  return values;
}

template <typename KeyType, typename ValueType, class KeyComparator>
template <typename Key, typename Value, typename Compare>
void BWTree<KeyType, ValueType, KeyComparator>::consolidateModifications(
    std::vector<std::pair<Key, Value>>& insert_records,
    std::set<std::pair<Key, Value>, Compare>& delete_records,
    typename std::vector<std::pair<Key, Value>>::iterator data_start,
    typename std::vector<std::pair<Key, Value>>::iterator data_end,
    Compare& less_fn, std::vector<std::pair<Key, Value>>& output_data) {
  // Perform set difference
  size_t begin_size = output_data.size();
  for (auto it = data_start; it != data_end; ++it) {
    auto del_it = delete_records.find(*it);
    if (del_it == delete_records.end()) {
      output_data.push_back(*it);
    }
  }
  // Add insert elements
  size_t middle_size = output_data.size();
  output_data.insert(output_data.end(), insert_records.begin(),
                     insert_records.end());
  auto begin_it = output_data.begin() + begin_size;
  auto middle_it = output_data.begin() + middle_size;
  // Perform merge
  std::inplace_merge(begin_it, middle_it, output_data.end(), less_fn);
}

template <typename KeyType, typename ValueType, class KeyComparator>
void BWTree<KeyType, ValueType, KeyComparator>::traverseAndConsolidateLeaf(
    BWNode* original_node, std::vector<BWNode*>& garbage_nodes,
    std::vector<std::pair<KeyType, ValueType>>& data, PID& sibling,
    bool& has_merge, BWNode*& merge_node, KeyType& lower_bound,
    KeyType& upper_bound) {
  using LessFnT = LessFn<KeyType, ValueType, KeyComparator>;

  LessFnT less_fn(m_key_less);
  std::vector<std::pair<KeyType, ValueType>> insert_records;
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
        // First check if we pass the split
        if (!has_split ||
            key_less(insert_node->ins_record.first, split_separator_key)) {
          // If we have a delete for this record, don't add
          auto it = delete_records.find(insert_node->ins_record);
          if (it == delete_records.end()) {
            insert_records.push_back(insert_node->ins_record);
          }
        }
        break;
      }
      case deltaDelete: {
        BWDeltaDeleteNode* delete_node = static_cast<BWDeltaDeleteNode*>(node);
        // Don't need to check if we pass the split because extra deletes
        // won't cause an issue
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
  garbage_nodes.push_back(node);
  std::sort(insert_records.begin(), insert_records.end(), less_fn);

  // node is a leaf node
  BWLeafNode* leaf_node = static_cast<BWLeafNode*>(node);

  lower_bound = leaf_node->lower_bound;

  auto data_start = leaf_node->data.begin();
  typename std::vector<std::pair<KeyType, ValueType>>::iterator data_end;
  if (has_split) {
    // Change sibling pointer if we did a split
    sibling = new_sibling;
    upper_bound = split_separator_key;
    data_end = std::upper_bound(
        leaf_node->data.begin(), leaf_node->data.end(), split_separator_key,
        [=](const KeyType& l, const std::pair<KeyType, ValueType>& r)
            -> bool { return m_key_less(l, std::get<0>(r)); });
  } else {
    sibling = leaf_node->next;
    upper_bound = leaf_node->upper_bound;
    data_end = leaf_node->data.end();
  }

  consolidateModifications<KeyType, ValueType, LessFnT>(
      insert_records, delete_records, data_start, data_end, less_fn, data);
}

template <typename KeyType, typename ValueType, class KeyComparator>
bool BWTree<KeyType, ValueType, KeyComparator>::consolidateLeafNode(
    PID id, BWNode* pid_node) {
  // Keep track of nodes so we can garbage collect later
  BWNode* original_node = pid_node;

  std::vector<BWNode*> garbage_nodes;
  std::vector<std::pair<KeyType, ValueType>> data;
  PID sibling;
  bool has_merge = false;
  BWNode* merge_node = nullptr;
  KeyType lower_bound;
  KeyType upper_bound;
  traverseAndConsolidateLeaf(original_node, garbage_nodes, data, sibling,
                             has_merge, merge_node, lower_bound, upper_bound);
  if (has_merge) {
    BWNode* dummy_node;
    KeyType dummy_bound;
    traverseAndConsolidateLeaf(merge_node, garbage_nodes, data, sibling,
                               has_merge, dummy_node, dummy_bound, upper_bound);
    assert(!has_merge);
  }

  // Check size and insert split if needed
  BWNode* swap_node = nullptr;

  bool did_split = false;
  size_t data_size = data.size();
  bwt_printf("Consolidated data size: %lu\n", data_size);
  if (LEAF_NODE_SIZE_MAX < data_size) {
    bwt_printf("Data size greater than threshold, splitting...\n");
    // Find separator key by grabbing middle element
    auto middle_it = data.begin() + data_size / 2;
    KeyType separator_key = middle_it->first;
    // Place second half in other node
    BWLeafNode* upper_leaf_node =
        new BWLeafNode(separator_key, upper_bound, sibling);
    upper_leaf_node->data.insert(upper_leaf_node->data.end(), middle_it,
                                 data.end());
    // Install second node
    PID new_split_pid = installPage(upper_leaf_node);
    // Place first half in one node
    BWLeafNode* lower_leaf_node =
        new BWLeafNode(lower_bound, separator_key, new_split_pid);
    lower_leaf_node->data.insert(lower_leaf_node->data.end(), data.begin(),
                                 middle_it);
    // Create split record
    BWDeltaSplitNode* split_node = new BWDeltaSplitNode(
        lower_leaf_node, separator_key, new_split_pid, upper_bound);
    swap_node = split_node;
    did_split = true;
  } else {
    BWLeafNode* consolidated_node =
        new BWLeafNode(lower_bound, upper_bound, sibling);
    consolidated_node->data.swap(data);

    if (data_size < LEAF_NODE_SIZE_MIN) {
      bwt_printf("Data size less than threshold, placing remove node...\n");
      // Install a remove delta on top of the node
      BWDeltaRemoveNode* remove_node = new BWDeltaRemoveNode(consolidated_node);
      swap_node = remove_node;
    } else {
      swap_node = consolidated_node;
    }
  }

  bool result =
      mapping_table[id].compare_exchange_strong(original_node, swap_node);
  if (result) {
    // Succeeded, request garbage collection of processed nodes
    addGarbageNodes(garbage_nodes);
  } else {
    if (did_split) {
      // Cleanup the split node
    }
    // Failed, cleanup
    deleteDeltaChain(swap_node);
  }
  return result;
}

template <typename KeyType, typename ValueType, class KeyComparator>
void BWTree<KeyType, ValueType, KeyComparator>::traverseAndConsolidateInner(
    BWNode* original_node, std::vector<BWNode*>& garbage_nodes,
    std::vector<std::pair<KeyType, PID>>& separators, bool& has_merge,
    BWNode*& merge_node, KeyType& lower_bound, KeyType& upper_bound) {
  using LessFnT = LessFn<KeyType, PID, KeyComparator>;

  LessFnT less_fn(m_key_less);
  std::set<PID> insert_pids;
  std::vector<std::pair<KeyType, PID>> insert_separators;
  std::set<std::pair<KeyType, PID>, LessFnT> delete_separators(less_fn);

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
        if (!has_split || key_lessequal(insert_node->new_split_separator_key,
                                        split_separator_key)) {
          std::pair<KeyType, PID> ins_separator(
              insert_node->new_split_separator_key,
              insert_node->new_split_sibling);
          // If we have a delete for this record, don't add
          auto it = delete_separators.find(ins_separator);
          if (it == delete_separators.end()) {
            // Check for duplicates
            auto dup_it = insert_pids.find(ins_separator.second);
            if (dup_it == insert_pids.end()) {
              insert_pids.insert(ins_separator.second);
              insert_separators.push_back(ins_separator);
            }
          }
        }
        break;
      }
      case deltaIndexTermDelete: {
        BWDeltaIndexTermDeleteNode* delete_node =
            static_cast<BWDeltaIndexTermDeleteNode*>(node);
        delete_separators.insert(
            {delete_node->remove_node_low_key, delete_node->node_to_remove});
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
    assert(node != nullptr);
  }
  garbage_nodes.push_back(node);
  std::sort(insert_separators.begin(), insert_separators.end(), less_fn);

  BWInnerNode* inner_node = static_cast<BWInnerNode*>(node);

  lower_bound = inner_node->lower_bound;

  auto base_start = inner_node->separators.begin();
  typename std::vector<std::pair<KeyType, PID>>::iterator base_end;
  if (has_split) {
    upper_bound = split_separator_key;
    // Find end of separators if split
    base_end = std::upper_bound(
        inner_node->separators.begin(), inner_node->separators.end(),
        split_separator_key,
        [=](const KeyType& l, const std::pair<KeyType, PID>& r)
            -> bool { return m_key_less(l, std::get<0>(r)); });
  } else {
    upper_bound = inner_node->upper_bound;
    base_end = inner_node->separators.end();
  }

  consolidateModifications<KeyType, PID, LessFnT>(
      insert_separators, delete_separators, base_start, base_end, less_fn,
      separators);
}

template <typename KeyType, typename ValueType, class KeyComparator>
bool BWTree<KeyType, ValueType, KeyComparator>::consolidateInnerNode(
    PID id, BWNode* pid_node) {
  BWNode* original_node = pid_node;

  // Keep track of nodes so we can garbage collect later
  std::vector<BWNode*> garbage_nodes;
  std::vector<std::pair<KeyType, PID>> separators;
  bool has_merge = false;
  BWNode* merge_node = nullptr;
  KeyType lower_bound;
  KeyType upper_bound;
  traverseAndConsolidateInner(original_node, garbage_nodes, separators,
                              has_merge, merge_node, lower_bound, upper_bound);
  if (has_merge) {
    BWNode* dummy_node;
    KeyType dummy_bound;
    traverseAndConsolidateInner(merge_node, garbage_nodes, separators,
                                has_merge, dummy_node, dummy_bound,
                                upper_bound);
    assert(!has_merge);
  }

  // Check size and insert split if needed
  BWNode* swap_node = nullptr;

  bool did_split = false;
  size_t data_size = separators.size();
  bwt_printf("Consolidated data size: %lu\n", data_size);
  if (INNER_NODE_SIZE_MAX < data_size) {
    bwt_printf("Data size greater than threshold, splitting...\n");
    // Find separator key by grabbing middle element
    auto middle_it = separators.begin() + data_size / 2;
    KeyType separator_key = middle_it->first;
    // Place first half in one node
    BWInnerNode* lower_inner_node = new BWInnerNode(lower_bound, separator_key);
    lower_inner_node->separators.insert(lower_inner_node->separators.end(),
                                        separators.begin(),
                                        middle_it);
    // Place second half in other node
    BWInnerNode* upper_inner_node = new BWInnerNode(separator_key, upper_bound);
    upper_inner_node->separators.insert(upper_inner_node->separators.end(),
                                        middle_it,
                                        separators.end());
    // Install second node
    PID new_split_pid = installPage(upper_inner_node);
    // Create split record
    BWDeltaSplitNode* split_node = new BWDeltaSplitNode(
        lower_inner_node, separator_key, new_split_pid, upper_bound);
    swap_node = split_node;
    did_split = true;
  } else {
    BWInnerNode* consolidated_node = new BWInnerNode(lower_bound, upper_bound);
    consolidated_node->separators.swap(separators);

    if (data_size < INNER_NODE_SIZE_MIN) {
      bwt_printf("Data size less than threshold, placing remove node...\n");
      // Install a remove delta on top of the node
      BWDeltaRemoveNode* remove_node = new BWDeltaRemoveNode(consolidated_node);
      swap_node = remove_node;
    } else {
      swap_node = consolidated_node;
    }
  }

  bool result = mapping_table[id].compare_exchange_strong(original_node,
                                                          swap_node);
  if (result) {
    // Succeeded, request garbage collection of processed nodes
    addGarbageNodes(garbage_nodes);
  } else {
    if (did_split) {
    }
    // Failed, cleanup
    deleteDeltaChain(swap_node);
  }
  return true;
}

template <typename KeyType, typename ValueType, class KeyComparator>
bool BWTree<KeyType, ValueType, KeyComparator>::isLeaf(BWNode* node) {
  BWNode* current_node = node;
  bool is_leaf = false;
  while (current_node != nullptr) {
    switch (current_node->type) {
    case PageType::deltaInsert:
    case PageType::deltaDelete:
    case PageType::leaf:
      is_leaf = true;
      current_node = nullptr;
      break;
    case PageType::deltaIndexTermInsert:
    case PageType::deltaIndexTermDelete:
    case PageType::inner:
      is_leaf = false;
      current_node = nullptr;
      break;
    default:
      BWDeltaNode* delta = static_cast<BWDeltaNode*>(current_node);
      current_node = delta->child_node;
      break;
    }
  }
  return is_leaf;
}

template <typename KeyType, typename ValueType, class KeyComparator>
bool BWTree<KeyType, ValueType, KeyComparator>::performConsolidation(
  PID id, BWNode* node) {
  // Figure out if this is a leaf or inner node
  bool is_leaf = isLeaf(node);;
  if (is_leaf) {
    return consolidateLeafNode(id, node);
  } else {
    return consolidateInnerNode(id, node);
  }
}

// Returns the first page where the key can reside
// For insert and delete this means the page on which delta record can be added
// For search means the first page the cursor needs to be constructed on
template <typename KeyType, typename ValueType, class KeyComparator>
std::pair<typename BWTree<KeyType, ValueType, KeyComparator>::PID,
          typename BWTree<KeyType, ValueType, KeyComparator>::BWNode*>
BWTree<KeyType, ValueType, KeyComparator>::findLeafPage(const KeyType& key) {
  // Root should always have a valid pid
  assert(m_root != NONE_PID);

  PID curr_pid = m_root.load();
  BWNode* curr_pid_root_node = mapping_table[curr_pid].load();
  BWNode* curr_node = curr_pid_root_node;

  PID parent_pid = NONE_PID;
  BWNode* parent_pid_root_node = nullptr;
  int chain_length = 0;  // Length of delta chain, including current node

  // Trigger consolidation

  // Trigger structure modifying operations
  // split, remove, merge
  bool still_searching = true;
  while (still_searching) {
    assert(curr_node != nullptr);
    chain_length += 1;

    if (DELTA_CHAIN_INNER_THRESHOLD < chain_length) {
      bwt_printf(
          "Delta chain greater than threshold, performing "
          "consolidation...\n");
      performConsolidation(curr_pid, curr_pid_root_node);
      // Reset to top of chain
      curr_pid_root_node = mapping_table[curr_pid].load();
      curr_node = curr_pid_root_node;
      chain_length = 0;
      continue;
    }

    // Set by any delta node which wishes to traverse to a child
    bool request_traverse_child = false;
    bool request_traverse_split = false;
    // Set when posting to update index fails due to change in parent
    bool request_restart_top = false;
    PID child_pid = NONE_PID;

    switch (curr_node->type) {
      ////////////////////////////////////////////////////////////////////////////
      /// Index Term Insert
      case PageType::deltaIndexTermInsert: {
        bwt_printf("Traversing index term insert node\n");
        BWDeltaIndexTermInsertNode* index_insert_delta =
            static_cast<BWDeltaIndexTermInsertNode*>(curr_node);
        if (key_greater(key, index_insert_delta->new_split_separator_key) &&
            key_lessequal(key, index_insert_delta->next_separator_key)) {
          // Shortcut to child page
          request_traverse_child = true;
          child_pid = index_insert_delta->new_split_sibling;
        } else {
          // Keep going down chain
          curr_node = index_insert_delta->child_node;
        }
        break;
      }
      ////////////////////////////////////////////////////////////////////////////
      /// Index Term Delete
      case PageType::deltaIndexTermDelete: {
        bwt_printf("Traversing index term delete node\n");
        BWDeltaIndexTermDeleteNode* index_delete_delta =
            static_cast<BWDeltaIndexTermDeleteNode*>(curr_node);
        if (key_greater(key, index_delete_delta->merge_node_low_key) &&
            key_lessequal(key, index_delete_delta->next_separator_key)) {
          // Shortcut to child page
          request_traverse_child = true;
          child_pid = index_delete_delta->node_to_merge_into;
        } else {
          // Keep going down chain
          curr_node = index_delete_delta->child_node;
        }
        break;
      }
      ////////////////////////////////////////////////////////////////////////////
      /// Inner
      case PageType::inner: {
        bwt_printf("Traversing inner node\n");
        BWInnerNode* inner_node = static_cast<BWInnerNode*>(curr_node);
        // The consolidate has to ensure that it does not leave empty
        // inner nodes
        assert(inner_node->separators.size() > 0);

        // TODO Change this to binary search
        PID next_pid = inner_node->separators.back().second;
        for (int i = 1; i < inner_node->separators.size(); i++) {
          bwt_printf("Inside for loop, i = %d\n", i);
          if (!key_less(inner_node->separators[i].first, key)) {
            next_pid = inner_node->separators[i - 1].second;
            break;
          }
        }

        request_traverse_child = true;
        child_pid = next_pid;
        break;
      }
      ////////////////////////////////////////////////////////////////////////////
      /// Leaf Insert
      case PageType::deltaInsert: {
        bwt_printf("Traversing insert node\n");
        BWDeltaInsertNode* insert_node =
            static_cast<BWDeltaInsertNode*>(curr_node);
        curr_node = insert_node->child_node;
        assert(curr_node != nullptr);
        break;
      }
      ////////////////////////////////////////////////////////////////////////////
      /// Leaf Delete
      case PageType::deltaDelete: {
        bwt_printf("Traversing delete node\n");
        BWDeltaDeleteNode* delete_node =
            static_cast<BWDeltaDeleteNode*>(curr_node);
        curr_node = delete_node->child_node;
        assert(curr_node != nullptr);
        break;
      }
      ////////////////////////////////////////////////////////////////////////////
      /// Leaf
      case PageType::leaf: {
        bwt_printf("Traversing leaf node\n");

        BWLeafNode* leaf_node = static_cast<BWLeafNode*>(curr_node);

        bwt_printf("leaf_node_size = %lu\n", leaf_node->data.size());

        // Check that we have not ended up on the wrong page for the key
        bool geq = key_greaterequal(key, leaf_node->lower_bound);
        bool le = key_lessequal(key, leaf_node->upper_bound);
        bwt_printf("key_greaterequal = %d\n", geq);
        bwt_printf("key_leq = %d\n", le);
        //assert(geq && le);
        still_searching = false;
        break;
      }
      ////////////////////////////////////////////////////////////////////////////
      /// Split
      case PageType::deltaSplit: {
        bwt_printf("Traversing split node\n");
        // Our invariant is that there should be no delta chains on top of a
        // split node
        assert(chain_length == 1);
        BWDeltaSplitNode* split_delta =
            static_cast<BWDeltaSplitNode*>(curr_node);

        // Install an IndexTermDeltaInsert and retry till it succeds. Thread
        // cannot proceed until this succeeds
        // 1) This might install multiple updates which say the same thing. The
        //    consolidate must be able to handle the duplicates
        // 2) The parent_pid might not be correct because the parent might have
        //    merged into some other node. This needs to be detected by the
        //    installIndexTermDeltaInsert and return the install_node_invalid
        //    which triggers a search from the root.

        // Must handle the case where the parent_pid is NONE_PID
        // Attempt to create a new inner node
        InstallDeltaResult status = install_try_again;
        if (parent_pid == NONE_PID) {
          // Create new root inner ndoe
          BWInnerNode* new_root = new BWInnerNode(KeyType::NEG_INF_KEY,
                                                  KeyType::POS_INF_KEY);
          // Add current root as child and new split sibling
          new_root->separators.emplace_back(KeyType::NEG_INF_KEY, curr_pid);
          new_root->separators.emplace_back(split_delta->separator_key,
                                            split_delta->split_sibling);
          // Install the new root
          PID new_root_pid = installPage(new_root);
          // Try to install the root
          bool result = m_root.compare_exchange_strong(curr_pid, new_root_pid);
          if (result) {
            bwt_printf("Replaced new root successfully.");
            status = install_success;
          } else {
            // Should cleanup the root page but for now this will be cleaned up
            // in the destructor
            bwt_printf("Compare exchange with root failed, restarting...");
            status = install_node_invalid;
            request_restart_top = true;
          }
        } else {
          while (status != install_success) {
            status = installIndexTermDeltaInsert(parent_pid,
                                                 parent_pid_root_node,
                                                 split_delta);
            if (status == install_need_consolidate) {
              bwt_printf("Split index parent needs consolidation, "
                         "performing...\n");
              performConsolidation(parent_pid, parent_pid_root_node);
            } else if (status == install_node_invalid) {
              bwt_printf("Split index parent invalid, restarting search...\n");
              // Restart from the top
              request_restart_top = true;
              break;
            }
            // Either success or try again, either way need to reload pid
            // Maybe we need to restar the whole process at this point?
            parent_pid_root_node = mapping_table[parent_pid].load();
          }
        }

        if (status == install_success) {
          bwt_printf("Split index parent install insert success\n");
          if (!key_equal(split_delta->separator_key,
                         split_delta->next_separator_key) &&
              key_greaterequal(key, split_delta->separator_key) &&
              key_lessequal(key, split_delta->next_separator_key)) {
            request_traverse_split = true;
            child_pid = split_delta->split_sibling;
          } else {
            curr_node = split_delta->child_node;
          }
        }
        break;
      }
      ////////////////////////////////////////////////////////////////////////////
      /// Remove
      case PageType::deltaRemove: {
        bwt_printf("Traversing remove node\n");
        // Note: should not trigger a remove on the left most leaf even if the
        // number of tuples is below a threshold

        // TODO: Install an installDeltaMerge on sibling and retry till it
        // succeds. Thread
        // cannot proceed until this succeeds
        assert(false);
      }
      ////////////////////////////////////////////////////////////////////////////
      /// Merge
      case PageType::deltaMerge: {
        bwt_printf("Traversing merge node\n");
        // Our invariant is that there should be no delta chains on top of a
        // merge node
        assert(chain_length == 1);
        BWDeltaMergeNode* merge_delta =
            static_cast<BWDeltaMergeNode*>(curr_node);

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
          // status = installIndexTermDeltaDelete(parent_pid, parent_pid_root_node,
          //                                      merge_delta);
          if (status == install_need_consolidate) {
            performConsolidation(parent_pid, parent_pid_root_node);
          } else if (status == install_node_invalid) {
            // Restart from the top
            request_restart_top = true;
            break;
          }
          parent_pid_root_node = mapping_table[parent_pid].load();
        }

        if (status == install_success) {
          if (key_greaterequal(key, merge_delta->separator_key)) {
            curr_node = merge_delta->merge_node;
          } else {
            curr_node = merge_delta->child_node;
          }
        }

        break;
      }
      default:
        assert(false);
    }

    if (request_traverse_split) {
      bwt_printf("Request to traverse to split PID %lu\n", child_pid);
      curr_pid = child_pid;
      curr_pid_root_node = mapping_table[curr_pid].load();
      curr_node = curr_pid_root_node;
      chain_length = 0;
    }

    if (request_traverse_child) {
      bwt_printf("Request to traverse to child PID %lu\n", child_pid);
      parent_pid = curr_pid;
      parent_pid_root_node = curr_pid_root_node;
      curr_pid = child_pid;
      curr_pid_root_node = mapping_table[curr_pid].load();
      curr_node = curr_pid_root_node;
      chain_length = 0;
    }

    if (request_restart_top) {
      bwt_printf("Request to restart from top %lu\n", curr_pid);
      parent_pid = NONE_PID;
      parent_pid_root_node = nullptr;
      curr_pid = m_root.load();
      curr_pid_root_node = mapping_table[curr_pid].load();
      curr_node = curr_pid_root_node;
      chain_length = 0;
    }
  }  // while(1)
  bwt_printf("Finished findLeafPage with PID %lu\n", curr_pid);

  return {curr_pid, curr_pid_root_node};
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
    PID leaf_pid, BWNode* node, const KeyType& key, const ValueType& value) {

  bool cas_success;
  auto ins_record = std::pair<KeyType, ValueType>(key, value);

  BWNode* old_leaf_p = node;

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
    PID leaf_pid, BWNode* node, const KeyType& key, const ValueType& value) {

  bool cas_success;
  auto delete_record = std::pair<KeyType, ValueType>(key, value);

  BWNode* old_leaf_p = node;

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
    PID node, BWNode* inner_node, const BWDeltaSplitNode* split_node) {
  BWNode* old_inner_p = inner_node;

  if (isSMO(old_inner_p)) {
    return install_need_consolidate;
  }

  KeyType new_separator_key = split_node->separator_key;
  PID split_sibling = split_node->split_sibling;
  KeyType next_separator_key = split_node->next_separator_key;
  BWNode* new_inner_p =
    (BWNode*)new BWDeltaIndexTermInsertNode(old_inner_p, new_separator_key,
                                            split_sibling, next_separator_key);

  bool cas_success =
      mapping_table[node].compare_exchange_strong(old_inner_p, new_inner_p);
  if (cas_success == false) {
    delete new_inner_p;
    return install_try_again;
  }

  return install_success;
}

template <typename KeyType, typename ValueType, typename KeyComparator>
typename BWTree<KeyType, ValueType, KeyComparator>::InstallDeltaResult
BWTree<KeyType, ValueType, KeyComparator>::installIndexTermDeltaDelete(
    __attribute__((unused)) PID node,
    __attribute__((unused)) BWNode* nner_node,
    __attribute__((unused)) const KeyType& key,
    __attribute__((unused)) const ValueType& value) {
  return install_try_again;
}

template <typename KeyType, typename ValueType, typename KeyComparator>
typename BWTree<KeyType, ValueType, KeyComparator>::InstallDeltaResult
BWTree<KeyType, ValueType, KeyComparator>::installDeltaMerge(
    __attribute__((unused)) PID node, __attribute__((unused)) PID sibling) {
  return install_try_again;
}

template <typename KeyType, typename ValueType, typename KeyComparator>
void BWTree<KeyType, ValueType, KeyComparator>::deleteDeltaChain(BWNode* node) {
  while (node != nullptr) {
    switch (node->type) {
      case PageType::deltaInsert:
        bwt_printf("Freeing insert node\n");
        break;
      case PageType::deltaDelete:
        bwt_printf("Freeing delete node\n");
        break;
      case PageType::deltaIndexTermInsert:
        bwt_printf("Freeing index insert node\n");
        break;
      case PageType::deltaIndexTermDelete:
        bwt_printf("Freeing index delete node\n");
        break;
      case PageType::deltaSplit:
        bwt_printf("Freeing split node\n");
        break;
      case PageType::deltaRemove:
        bwt_printf("Freeing remove node\n");
        break;
      case PageType::deltaMerge:
        bwt_printf("Freeing merge node\n");
        break;
      case PageType::inner:
        bwt_printf("Freeing inner node\n");
        break;
      case PageType::leaf:
        bwt_printf("Freeing leaf node\n");
        break;
      default:
        assert(false);
    }

    switch (node->type) {
      case PageType::deltaInsert:
      case PageType::deltaDelete:
      case PageType::deltaIndexTermInsert:
      case PageType::deltaIndexTermDelete:
      case PageType::deltaSplit: {
        BWDeltaNode* delta = static_cast<BWDeltaNode*>(node);
        node = delta->child_node;
        delete delta;
        break;
      }
      case PageType::deltaRemove: {
        delete node;
        node = nullptr;
        break;
      }
      case PageType::deltaMerge: {
        BWDeltaMergeNode* merge_node = static_cast<BWDeltaMergeNode*>(node);
        deleteDeltaChain(merge_node->merge_node);
        node = merge_node->child_node;
        delete merge_node;
        break;
      }
      case PageType::inner:
      case PageType::leaf: {
        delete node;
        node = nullptr;
        break;
      }
      default:
        assert(false);
    }
  }
}

template <typename KeyType, typename ValueType, typename KeyComparator>
void BWTree<KeyType, ValueType, KeyComparator>::addGarbageNodes(
    std::vector<BWNode*>& garbage) {
  while (!garbage_mutex.try_lock());
  garbage_nodes.insert(garbage_nodes.end(), garbage.begin(), garbage.end());
  garbage_mutex.unlock();
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
