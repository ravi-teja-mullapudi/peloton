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
#include <unordered_map>
#include <mutex>
#include <string>
#include <iostream>
#include "backend/storage/tuple.h"
#include "backend/common/logger.h"

//#define BWTREE_DEBUG

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

#define bwt_printf(fmt, ...)   \
  do {                         \
    dummy(fmt, ##__VA_ARGS__); \
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

static void dummy(const char*, ...) {}

template <typename KeyType, typename ValueType, typename KeyComparator>
struct LessFnT;
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

template <typename KeyType, typename ValueType, typename KeyComparator>
class BWTree {
  class InteractiveDebugger;
  friend InteractiveDebugger;

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
    std::vector<std::pair<KeyType, std::vector<ValueType>>> data;
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

  /*
   * class ConsistencyChecker - Checks for tree structural integrity
   *
   * NOTE: Only prototypes are defined. We need to come up with more
   * checks
   */
  class ConsistencyChecker {
   public:
    ConsistencyChecker() {}
    void printTreeStructure();
    bool checkInnerNode(BWNode* node_p);
    bool checkLeafNode(BWNode* node_p);
    bool checkSeparator(BWInnerNode* inner_node_p);
    bool checkInnerNodeBound(BWNode* node_p);
    bool checkLeafNodeBound(BWNode* node_p);
  };

  /*
   * class EpochRecord - We keep such a record for each epoch
   */
  struct EpochRecord {
    uint64_t thread_count;
    std::vector<BWNode*> node_list;

    EpochRecord(EpochRecord&& et) {
      node_list = std::move(et.node_list);
      thread_count = et.thread_count;

      return;
    }

    EpochRecord& operator=(EpochRecord&& et) {
      node_list = std::move(et.node_list);
      thread_count = et.thread_count;

      return *this;
    }

    EpochRecord() : thread_count(1) {}
  };

  /*
   * EpochManager - Manages epoch and garbage collection
   *
   * NOTE: We implement this using std::mutex to handle std::vector
   * and std::unordered_map
   */
  class EpochManager {
   private:
    using bw_epoch_t = uint64_t;
    // This could be handled with CAS
    std::atomic<bw_epoch_t> current_epoch;
    std::mutex lock;

    // It is a counter that records how many joins has been called;
    // If this reaches a threshold then we just start next epoch
    // synchronously inside some thread's join() procedure
    uint64_t join_counter;

    // We allow at most 1000 join threads inside one epoch
    // This could be tuned
    static const uint64_t join_threshold = 1000;

    // This structure must be handled inside critical section
    // Also we rely on the fact that when we scan the map, epochs are
    // scanned in an increasing order which facilitates our job
    std::map<bw_epoch_t, EpochRecord> garbage_list;

   public:
    EpochManager();

    // Called by the thread to announce their existence
    bw_epoch_t joinEpoch();
    void leaveEpoch(bw_epoch_t e);
    void advanceEpoch();

    void addGarbageNode(BWNode* node_p);

    void sweepAndClean();
  };

  /*
   * InteractiveDebugger - Allows us to explore the tree interactively
   */
  class InteractiveDebugger {
   public:
    BWTree* tree;

    PID current_pid;
    BWNode* current_node_p;
    PageType current_type;

    std::stack<BWNode*> node_stack;

    void printPrompt() {
      std::cout << "[PID=" << current_pid << "]>>";
      return;
    }

    void prepareNode(BWNode* node_p) {
      // Node pointer must be valid
      assert(node_p != nullptr);
      current_node_p = node_p;
      current_type = node_p->type;

      return;
    }

    /*
     * prepareNodeByPID() - Returns false if the PID is invalid
     */
    bool prepareNodeByPID(PID pid) {
      if (pid == NONE_PID) {
        return false;
      }

      current_pid = pid;
      prepareNode(tree->mapping_table[pid].load());

      return true;
    }

    std::string pageTypeToString(PageType type) {
      switch (type) {
        case leaf:
          return "Leaf";
        case inner:
          return "Inner";
        case deltaInsert:
          return "Delta Insert";
        case deltaDelete:
          return "Delta Delete";
        case deltaSplit:
          return "Delta Split";
        case deltaIndexTermInsert:
          return "Index Insert";
        case deltaIndexTermDelete:
          return "Index Delete";
        case deltaRemove:
          return "Remove";
        case deltaMerge:
          return "Merge";
        default:
          return "Unknown Type (Error!)";
      }

      assert(false);
      return "";
    }

    void processPrint() {
      std::string s;
      std::cin >> s;

      if (s == "") {
        std::cout << "Nothing to print!" << std::endl;
        return;
      } else if (s == "node-pointer") {
        std::cout << current_node_p << std::endl;
      } else if (s == "type") {
        std::cout << current_type << " (" << pageTypeToString(current_type)
                  << ")" << std::endl;
      } else {
        std::cout << "Unkown print argument: " << s << std::endl;
      }

      return;
    }

    void processGotoChild() {
      BWDeltaNode* delta_node_p = nullptr;

      switch (current_node_p->type) {
        case deltaDelete:
        case deltaIndexTermDelete:
        case deltaIndexTermInsert:
        case deltaInsert:
        case deltaMerge:
        case deltaRemove:
        case deltaSplit:
          delta_node_p = (static_cast<BWDeltaNode*>(current_node_p));
          // Push current node into the stack to facilitate back tracking
          node_stack.push(current_node_p);
          // Make this current node
          prepareNode(static_cast<BWNode*>(delta_node_p->child_node));
          break;
        case leaf:
        case inner:
          std::cout << "Type (" << pageTypeToString(current_type)
                    << ") does not have child node" << std::endl;
          break;
      }

      return;
    }

    void start() {
      // We could not start with empty root node
      assert(prepareNodeByPID(tree->m_root.load()) == true);

      std::cout << "********* Interactive Debugger *********\n";

      while (1) {
        printPrompt();

        std::string opcode;
        std::cin >> opcode;

        // If read EOF then resume BWTree execution
        if (!std::cin) {
          return;
        }

        if (opcode == "exit") {
          exit(0);
        } else if (opcode == "print") {
          processPrint();
        } else if (opcode == "type") {
          std::cout << current_type << " (" << pageTypeToString(current_type)
                    << ")" << std::endl;
        } else if (opcode == "goto-child") {
          processGotoChild();
        } else {
          std::cout << "Unknown command: " << opcode << std::endl;
        }
      }

      return;
    }

    InteractiveDebugger(BWTree* _tree)
        : tree(_tree), current_pid(0), current_node_p(nullptr) {}
  };

  /// //////////////////////////////////////////////////////////////
  /// Method decarations & definitions
  /// //////////////////////////////////////////////////////////////
 public:
  // TODO: pass a settings structure as we go along instead of
  // passing in individual parameter values
  BWTree(KeyComparator _m_key_less, bool _m_unique_keys);
  ~BWTree();

  bool insert(const KeyType& key, const ValueType& value);
  bool exists(const KeyType& key);
  bool erase(const KeyType& key, const ValueType& value);

  size_t getMemoryFootprint() { return m_foot_print; }

  std::vector<ValueType> find(const KeyType& key);
  // Scan all values
  void getAllValues(std::vector<ValueType>& result);

 private:
  std::vector<ValueType> collectPageContentsByKey(BWNode* node_p,
                                                  const KeyType& key,
                                                  PID& next_page,
                                                  KeyType& high_key);
  std::vector<std::pair<KeyType, std::vector<ValueType>>>
  collectAllPageContents(BWNode* node_p, PID& next_page, KeyType& high_key);

  /*
   * isValueEqual() - Compare two values and see if they are equal
   */
  inline bool isValueEqual(const ValueType& a, const ValueType& b) const {
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
  void consolidateModificationsLeaf(
      std::vector<std::pair<KeyType, ValueType>>& insert_records,
      std::set<std::pair<KeyType, ValueType>,
               LessFn<KeyType, ValueType, KeyComparator>>& delete_records,
      typename std::vector<std::pair<KeyType, std::vector<ValueType>>>::iterator
          data_start,
      typename std::vector<std::pair<KeyType, std::vector<ValueType>>>::iterator
          data_end,
      LessFn<KeyType, ValueType, KeyComparator>& less_fn,
      std::vector<std::pair<KeyType, std::vector<ValueType>>>& output_data);

  void consolidateModificationsInner(
      std::vector<std::pair<KeyType, PID>>& insert_records,
      std::set<std::pair<KeyType, PID>, LessFn<KeyType, PID, KeyComparator>>&
          delete_records,
      typename std::vector<std::pair<KeyType, PID>>::iterator data_start,
      typename std::vector<std::pair<KeyType, PID>>::iterator data_end,
      LessFn<KeyType, PID, KeyComparator>& less_fn,
      std::vector<std::pair<KeyType, PID>>& output_data);

  // Internal functions to be implemented
  void traverseAndConsolidateLeaf(
      BWNode* node, std::vector<BWNode*>& garbage_nodes,
      std::vector<std::pair<KeyType, std::vector<ValueType>>>& data,
      PID& sibling, bool& has_merge, BWNode*& merge_node, KeyType& lower_bound,
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

  BWNode* spinOnSMOByKey(KeyType& key);

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

  std::atomic<size_t> m_foot_print;

  // Not efficient but just for correctness
  std::mutex garbage_mutex;
  std::vector<BWNode*> garbage_nodes;

  std::atomic<PID> m_root;
  const KeyComparator m_key_less;
  const bool m_unique_keys;
  const ValueComparator m_val_equal;

  // TODO: UNFINISHED!!!
  const ConsistencyChecker checker;

  // TODO: Add a global garbage vector per epoch using a lock
  EpochManager epoch_mgr;
  InteractiveDebugger idb;

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
BWTree<KeyType, ValueType, KeyComparator>::BWTree(KeyComparator _m_key_less,
                                                  bool _m_unique_keys)
    : current_mapping_table_size(0),
      next_pid(0),
      m_foot_print(0),
      m_key_less(_m_key_less),
      m_unique_keys(_m_unique_keys),
      m_val_equal(),
      checker(),
      epoch_mgr(),
      idb(this) {
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
 *   - deltaMerge
 *
 * Also asserts if the node is a remove since there is nothing to consolidate
 * in case of a remove
 */
template <typename KeyType, typename ValueType, typename KeyComparator>
bool BWTree<KeyType, ValueType, KeyComparator>::isSMO(BWNode* n) {
  PageType type = n->type;
  assert(type != PageType::deltaRemove);
  return (type == PageType::deltaSplit) || (type == PageType::deltaMerge);
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
 */
template <typename KeyType, typename ValueType, typename KeyComparator>
void BWTree<KeyType, ValueType, KeyComparator>::getAllValues(
    std::vector<ValueType>& result) {
  bwt_printf("Get All Values!\n");

  // This will find the left most leaf page in the tree
  BWNode* curr_node;
  PID start_page;
  std::tie(start_page, curr_node) = findLeafPage(KeyType::NEG_INF_KEY);

  while (curr_node != nullptr) {
    PID next_page;
    KeyType high_key;
    std::vector<std::pair<KeyType, std::vector<ValueType>>> curr_page_values =
        collectAllPageContents(curr_node, next_page, high_key);

    for (auto key_it = curr_page_values.begin();
         key_it != curr_page_values.end(); key_it++) {
      result.insert(result.end(), key_it->second.begin(), key_it->second.end());
    }

    // There is nothing more to check
    if (next_page == NONE_PID) {
      curr_node = nullptr;
    } else {
      curr_node = mapping_table[next_page];
    }
  }
}

/*
 * exists() - Return true if a tuple with the given key exists in the tree
 *
 */
template <typename KeyType, typename ValueType, typename KeyComparator>
bool BWTree<KeyType, ValueType, KeyComparator>::exists(const KeyType& key) {
  bwt_printf("key = %d\n", key);

  std::vector<ValueType> values = find(key);
  return values.size() > 0;
}

/*
 * collectPageContentsByKey() - Collect items in a page with a given key
 */
template <typename KeyType, typename ValueType, typename KeyComparator>
std::vector<ValueType>
BWTree<KeyType, ValueType, KeyComparator>::collectPageContentsByKey(
    BWNode* node_p, const KeyType& key, PID& next_page, KeyType& high_key) {
  std::vector<ValueType> values;

  std::vector<std::pair<KeyType, std::vector<ValueType>>> all_records =
      collectAllPageContents(node_p, next_page, high_key);

  // Filter tuple values by key
  for (auto it = all_records.begin(); it != all_records.end(); ++it) {
    if (key_equal(it->first, key)) {
      values.insert(values.begin(), it->second.begin(), it->second.end());
      break;
    }
  }

  return values;
}

/*
 * collectAllPageContents() - Collect items on a given logical page (PID)
 */
template <typename KeyType, typename ValueType, typename KeyComparator>
std::vector<std::pair<KeyType, std::vector<ValueType>>>
BWTree<KeyType, ValueType, KeyComparator>::collectAllPageContents(
    BWNode* node_p, PID& next_page, KeyType& high_key) {
  using LessFnT = LessFn<KeyType, ValueType, KeyComparator>;

  LessFnT less_fn(m_key_less);
  BWNode* curr_node = node_p;

  std::vector<std::pair<KeyType, std::vector<ValueType>>> all_records;
  std::set<std::pair<KeyType, ValueType>, LessFnT> delete_records(less_fn);
  std::vector<std::pair<KeyType, ValueType>> insert_records;

  while (curr_node != nullptr) {
    switch (curr_node->type) {
      case PageType::deltaRemove: {
        // TODO: Reason if this is the right thing to do
        BWDeltaRemoveNode* remove_node =
            static_cast<BWDeltaRemoveNode*>(curr_node);
        curr_node = remove_node->child_node;
        break;
      }
      case PageType::deltaSplit: {
        BWDeltaSplitNode* split_node =
            static_cast<BWDeltaSplitNode*>(curr_node);
        // The side pointer to the next page is not logically a part of the
        // page and is ignored since the sibling pointer will point to it. Any
        // client doing a scan or find will look in the sibiling to get its
        // contents if necessary
        curr_node = split_node->child_node;
        break;
      }
      case PageType::deltaMerge: {
        BWDeltaMergeNode* merge_delta =
            static_cast<BWDeltaMergeNode*>(curr_node);
        // Both the merge node and child node are considered part of the same
        // logical page so the contents of both the physical pages has to be
        // collected

        // Note: The next PID of the left node in the merge should point to no
        // other node. The sibling information is part of the merge node.
        PID left_next;
        KeyType left_high_key;
        std::vector<std::pair<KeyType, std::vector<ValueType>>> child_contents =
            collectAllPageContents(merge_delta->child_node, left_next,
                                   left_high_key);
        assert(left_next == NONE_PID);
        // Add all the collected values to the vector of records to be inserted
        for (auto it = child_contents.begin(); it != child_contents.end();
             ++it) {
          for (auto val_it = it->second.begin(); val_it != it->second.end();
               val_it++) {
            insert_records.push_back(std::make_pair(it->first, *val_it));
          }
        }
        curr_node = merge_delta->merge_node;
        break;
      }
      case PageType::deltaInsert: {
        BWDeltaInsertNode* node = static_cast<BWDeltaInsertNode*>(curr_node);

        // If the tuple is already in the delete list ignore it
        auto bounds = std::equal_range(
            delete_records.begin(), delete_records.end(),
            std::make_pair(node->ins_record.first, ValueType{}), less_fn);

        bool found = false;
        for (auto del_it = bounds.first; del_it != bounds.second; ++del_it) {
          if (m_val_equal(del_it->second, node->ins_record.second)) {
            found = true;
            break;
          }
        }

        if (!found) {
          insert_records.push_back(node->ins_record);
        }

        curr_node = node->child_node;
        break;
      }
      case PageType::deltaDelete: {
        BWDeltaDeleteNode* node = static_cast<BWDeltaDeleteNode*>(curr_node);
        delete_records.insert(node->del_record);
        curr_node = node->child_node;
        break;
      }
      case PageType::leaf: {
        BWLeafNode* node = static_cast<BWLeafNode*>(curr_node);

        for (auto it = node->data.begin(); it != node->data.end(); ++it) {
          // If the tuple is already in the delete list ignore it
          std::vector<ValueType> values;
          auto bounds =
              std::equal_range(delete_records.begin(), delete_records.end(),
                               std::make_pair(it->first, ValueType{}), less_fn);
          for (auto val_it = it->second.begin(); val_it < it->second.end();
               ++val_it) {
            bool found = false;
            for (auto del_it = bounds.first; del_it != bounds.second;
                 ++del_it) {
              if (m_val_equal(del_it->second, *val_it)) {
                found = true;
                break;
              }
            }
            if (!found) {
              values.push_back(*val_it);
            }
          }

          if (values.size() > 0) {
            all_records.push_back(std::make_pair(it->first, values));
          }
        }

        // Add all the insert records
        for (auto ins_it = insert_records.begin();
             ins_it != insert_records.end(); ins_it++) {
          bool found = false;
          for (auto out_it = all_records.begin(); out_it != all_records.end();
               out_it++) {
            if (key_equal(out_it->first, ins_it->first)) {
              out_it->second.push_back(ins_it->second);
              found = true;
              break;
            }
          }
          if (!found) {
            std::vector<ValueType> values{ins_it->second};
            all_records.push_back(std::make_pair(ins_it->first, values));
          }
        }

        curr_node = nullptr;
        next_page = node->next;
        high_key = node->upper_bound;
        break;
      }
      case PageType::deltaIndexTermInsert:
      case PageType::deltaIndexTermDelete:
      case PageType::inner: {
        // Could our PID have been replaced with an index node via a remove
        // and then subsequent split?
        // This should not happen currently because we do not reuse pid values
        // also the garbage collection should ensure that if some thread is
        // is reading the page it should not have been reused
        bwt_printf("The PID points to an index node\n");
        assert(false);
        break;
      }
      default:
        assert(false);
    }
  }
  return all_records;
}

/*
 * insert() - Insert a key-value pair into B-Tree
 *
 * NOTE: Natural duplicated key support - we do not check for
 * duplicated key/val pair since they are allowed to be there
 */
template <typename KeyType, typename ValueType, typename KeyComparator>
bool BWTree<KeyType, ValueType, KeyComparator>::insert(const KeyType& key,
                                                       const ValueType& value) {
  bool insert_success = false;
  m_foot_print += 4;
  if (m_unique_keys) {
    std::vector<ValueType> values = find(key);
    // There can only be one value corresponding to a key
    assert(values.size() <= 1);
    if (values.size() > 0) return insert_success;
  }

  do {
    // First reach the leaf page where the key should be inserted
    PID page_pid;
    BWNode* node;
    std::tie(page_pid, node) = findLeafPage(key);

    // Then install an insertion record
    InstallDeltaResult result = installDeltaInsert(page_pid, node, key, value);
    if (result == install_need_consolidate) {
      insert_success = false;
      // If consolidation fails then we know some other thread
      // has performed consolidation for us and we just try again
      performConsolidation(page_pid, node);
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
  m_foot_print -= 4;
  std::vector<ValueType> values = find(key);
  if (values.size() == 0) return delete_success;

  do {
    // First reach the leaf page where the key should be inserted
    PID page_pid;
    BWNode* node;
    std::tie(page_pid, node) = findLeafPage(key);

    // Then install an insertion record
    InstallDeltaResult result = installDeltaDelete(page_pid, node, key, value);
    if (result == install_need_consolidate) {
      delete_success = false;
      // If consolidation fails then we know some other thread
      // has performed consolidation for us and we just try again
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
  PID curr_page;
  BWNode* curr_node;

  // Find the leaf page the key maps into
  std::tie(curr_page, curr_node) = findLeafPage(key);
  PID next_page;
  KeyType high_key;
  std::vector<ValueType> values =
      collectPageContentsByKey(curr_node, key, next_page, high_key);

  // Check that the high_key of the leaf page is less than that of the
  // key being searched otherwise something has gone wrong
  assert(key_less(key, high_key));

  return values;
}

template <typename KeyType, typename ValueType, class KeyComparator>
void BWTree<KeyType, ValueType, KeyComparator>::consolidateModificationsLeaf(
    __attribute__((unused))
    std::vector<std::pair<KeyType, ValueType>>& insert_records,
    std::set<std::pair<KeyType, ValueType>,
             LessFn<KeyType, ValueType, KeyComparator>>& delete_records,
    typename std::vector<std::pair<KeyType, std::vector<ValueType>>>::iterator
        data_start,
    typename std::vector<std::pair<KeyType, std::vector<ValueType>>>::iterator
        data_end,
    LessFn<KeyType, ValueType, KeyComparator>& less_fn,
    std::vector<std::pair<KeyType, std::vector<ValueType>>>& output_data) {
  using sortFnT = LessFn<KeyType, std::vector<ValueType>, KeyComparator>;
  sortFnT sort_fn(m_key_less);

  for (auto it = data_start; it != data_end; ++it) {
    std::vector<ValueType> values;
    auto bounds =
        std::equal_range(delete_records.begin(), delete_records.end(),
                         std::make_pair(it->first, ValueType{}), less_fn);

    for (auto val_it = it->second.begin(); val_it != it->second.end();
         val_it++) {
      bool found = false;
      for (auto del_it = bounds.first; del_it != bounds.second; ++del_it) {
        if (m_val_equal(del_it->second, *val_it)) {
          found = true;
          break;
        }
      }
      if (!found) {
        values.push_back(*val_it);
      }
    }
    if (values.size() > 0) {
      output_data.push_back(std::make_pair(it->first, values));
    }
  }

  // Add insert elements
  // TODO: Naive way of doing the search can be optimized
  for (auto ins_it = insert_records.begin(); ins_it != insert_records.end();
       ins_it++) {
    bool found = false;
    for (auto out_it = output_data.begin(); out_it != output_data.end();
         out_it++) {
      if (key_equal(out_it->first, ins_it->first)) {
        out_it->second.push_back(ins_it->second);
        found = true;
        break;
      }
    }
    if (!found) {
      std::vector<ValueType> values{ins_it->second};
      output_data.push_back(std::make_pair(ins_it->first, values));
    }
  }

  // Sort the output
  std::sort(output_data.begin(), output_data.end(), sort_fn);
}

template <typename KeyType, typename ValueType, class KeyComparator>
void BWTree<KeyType, ValueType, KeyComparator>::consolidateModificationsInner(
    std::vector<std::pair<KeyType, PID>>& insert_records,
    std::set<std::pair<KeyType, PID>, LessFn<KeyType, PID, KeyComparator>>&
        delete_records,
    typename std::vector<std::pair<KeyType, PID>>::iterator data_start,
    typename std::vector<std::pair<KeyType, PID>>::iterator data_end,
    LessFn<KeyType, PID, KeyComparator>& less_fn,
    std::vector<std::pair<KeyType, PID>>& output_data) {
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
    std::vector<std::pair<KeyType, std::vector<ValueType>>>& data, PID& sibling,
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
          auto bounds = std::equal_range(
              delete_records.begin(), delete_records.end(),
              std::make_pair(insert_node->ins_record.first, ValueType{}),
              less_fn);

          bool found = false;
          for (auto del_it = bounds.first; del_it != bounds.second; ++del_it) {
            if (m_val_equal(del_it->second, insert_node->ins_record.second)) {
              found = true;
              break;
            }
          }

          if (!found) {
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

  if (has_split) {
    // Change sibling pointer if we did a split
    sibling = new_sibling;
    upper_bound = split_separator_key;
  } else {
    sibling = leaf_node->next;
    upper_bound = leaf_node->upper_bound;
  }

  consolidateModificationsLeaf(insert_records, delete_records,
                               leaf_node->data.begin(), leaf_node->data.end(),
                               less_fn, data);
}

template <typename KeyType, typename ValueType, class KeyComparator>
bool BWTree<KeyType, ValueType, KeyComparator>::consolidateLeafNode(
    PID id, BWNode* pid_node) {
  // Keep track of nodes so we can garbage collect later
  BWNode* original_node = pid_node;

  std::vector<BWNode*> garbage_nodes;
  std::vector<std::pair<KeyType, std::vector<ValueType>>> data;
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

  if (has_split) {
    upper_bound = split_separator_key;
  } else {
    upper_bound = inner_node->upper_bound;
  }

  consolidateModificationsInner(
      insert_separators, delete_separators, inner_node->separators.begin(),
      inner_node->separators.end(), less_fn, separators);
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
                                        separators.begin(), middle_it);
    // Place second half in other node
    BWInnerNode* upper_inner_node = new BWInnerNode(separator_key, upper_bound);
    upper_inner_node->separators.insert(upper_inner_node->separators.end(),
                                        middle_it, separators.end());
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

  bool result =
      mapping_table[id].compare_exchange_strong(original_node, swap_node);
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
  bool is_leaf = isLeaf(node);
  if (is_leaf) {
    return consolidateLeafNode(id, node);
  } else {
    return consolidateInnerNode(id, node);
  }
}

// Returns the first page where the key can reside
// For insert and delete this means the page on which delta record can be
// added
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
        if (key_greaterequal(key,
                             index_insert_delta->new_split_separator_key) &&
            key_less(key, index_insert_delta->next_separator_key)) {
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
        if (key_greaterequal(key, index_delete_delta->merge_node_low_key) &&
            key_less(key, index_delete_delta->next_separator_key)) {
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
        bool le = key_less(key, leaf_node->upper_bound);
        bwt_printf("key_greaterequal = %d\n", geq);
        bwt_printf("key_leq = %d\n", le);
        assert(geq && le);
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
        // 1) This might install multiple updates which say the same thing.
        // The
        //    consolidate must be able to handle the duplicates
        // 2) The parent_pid might not be correct because the parent might
        // have
        //    merged into some other node. This needs to be detected by the
        //    installIndexTermDeltaInsert and return the
        //    install_node_invalid
        //    which triggers a search from the root.

        // Must handle the case where the parent_pid is NONE_PID
        // Attempt to create a new inner node
        InstallDeltaResult status = install_try_again;
        if (parent_pid == NONE_PID) {
          // Create new root inner ndoe
          BWInnerNode* new_root =
              new BWInnerNode(KeyType::NEG_INF_KEY, KeyType::POS_INF_KEY);
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
            // Should cleanup the root page but for now this will be cleaned
            // up
            // in the destructor
            bwt_printf("Compare exchange with root failed, restarting...");
            status = install_node_invalid;
            request_restart_top = true;
          }
        } else {
          while (status != install_success) {
            status = installIndexTermDeltaInsert(
                parent_pid, parent_pid_root_node, split_delta);
            if (status == install_need_consolidate) {
              bwt_printf(
                  "Split index parent needs consolidation, "
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
          if (key_greaterequal(key, split_delta->separator_key) &&
              key_less(key, split_delta->next_separator_key)) {
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
        // Note: should not trigger a remove on the left most leaf even if
        // the
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
        // 1) This might install multiple updates which say the same thing.
        // The
        //    consolidate must be able to handle the duplicates
        // 2) The parent_pid might not be correct because the parent might
        // have
        //    merged into some other node. This needs to be detected by the
        //    installIndexTermDeltaDelete and return the
        //    install_node_invalid
        //    which triggers a search from the root.
        InstallDeltaResult status = install_try_again;
        while (status != install_success) {
          // status = installIndexTermDeltaDelete(parent_pid,
          // parent_pid_root_node,
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

// This function will assign a page ID for a given page, and put that page
// into the mapping table
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
  BWNode* new_inner_p = (BWNode*)new BWDeltaIndexTermInsertNode(
      old_inner_p, new_separator_key, split_sibling, next_separator_key);

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
    __attribute__((unused)) PID node, __attribute__((unused)) BWNode* nner_node,
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
  while (!garbage_mutex.try_lock())
    ;
  garbage_nodes.insert(garbage_nodes.end(), garbage.begin(), garbage.end());
  garbage_mutex.unlock();
}

/////////////////////////////////////////////////////////////////////
// Epoch Manager Member Functions
/////////////////////////////////////////////////////////////////////

/*
 * EpochManager Constructor - Initialize current epoch to 0
 */
template <typename KeyType, typename ValueType, typename KeyComparator>
BWTree<KeyType, ValueType, KeyComparator>::BWTree::EpochManager::EpochManager()
    : current_epoch(0), join_counter(0) {
  return;
}

/*
 * advanceEpoch() - Advance to a new epoch. All older epoch garbages are now
 * pending to be collected once all older epoches has cleared
 */
template <typename KeyType, typename ValueType, typename KeyComparator>
void BWTree<KeyType, ValueType,
            KeyComparator>::BWTree::EpochManager::advanceEpoch() {
  bool retry = false;
  do {
    bw_epoch_t old_epoch = current_epoch.load();
    bw_epoch_t new_epoch = old_epoch++;
    retry = current_epoch.compare_exchange_strong(old_epoch, new_epoch);
  } while (retry == false);

  return;
}

/*
 * EpochManager::joinEpoch() - A thread joins the current epoch
 */
template <typename KeyType, typename ValueType, typename KeyComparator>
typename BWTree<KeyType, ValueType, KeyComparator>::EpochManager::bw_epoch_t
BWTree<KeyType, ValueType, KeyComparator>::EpochManager::joinEpoch() {
  lock.lock();
  // Critical section begin

  bw_epoch_t e = current_epoch.load();
  auto it = garbage_list.find(e);

  // This is the first thread that joins this epoch
  // Create a new record
  if (it == garbage_list.end()) {
    garbage_list[e] = EpochRecord{};
  } else {
    // Otherwise just increase its count
    EpochRecord er = std::move(it->second);
    er.thread_count++;
    garbage_list[e] = std::move(er);
  }

  // This does not need be done atomically
  join_counter++;
  if (join_counter > EpochManager::join_threshold) {
    advanceEpoch();
    join_counter = 0;
  }

  // Critical section end
  lock.unlock();
  return e;
}

/*
 * EpochManager::leaveEpoch() - Leave an epoch that a thread was in
 *
 * This function decreases the corrsponding epoch counter by 1, and if it
 * goes to 0, just remove all references inside that epoch since now we
 * know nobody could be referencing to the nodes
 */
template <typename KeyType, typename ValueType, typename KeyComparator>
void BWTree<KeyType, ValueType,
            KeyComparator>::BWTree::EpochManager::leaveEpoch(bw_epoch_t e) {
  lock.lock();

  auto it = garbage_list.find(e);
  assert(it != garbage_list.end());

  EpochRecord er = std::move(it->second);
  assert(er.thread_count > 0);
  er.thread_count--;

  bool need_clean = (er.thread_count == 0) && (current_epoch.load() != e);

  garbage_list[e] = std::move(er);

  if (need_clean == true) {
    sweepAndClean();
  }

  lock.unlock();
  return;
}

/*
 * EpochManager::sweepAndClean() - Cleans oldest epochs whose ref count is 0
 *
 * We never free memory for the current epoch (there might be a little bit
 *delay)
 * We stops scanning the list of epoches once an epoch whose
 * ref count != not 0 is seen
 *
 * NOTE: This function must be called under critical section
 */
template <typename KeyType, typename ValueType, typename KeyComparator>
void BWTree<KeyType, ValueType,
            KeyComparator>::BWTree::EpochManager::sweepAndClean() {
  // This could be a little bit late compared the real time epoch
  // but it is OK since we could recycle what we have missed in the next run
  bw_epoch_t e = current_epoch.load();

  for (auto it = garbage_list.begin(); it != garbage_list.end(); it++) {
    assert(it->first <= e);

    if (it->first == e) {
      break;
    } else if (it->second.thread_count > 0) {
      // We stop when some epoch still has ongoing threads
      break;
    }

    std::vector<BWNode*> node_list = std::move(it->second);

    for (auto it2 = node_list.begin(); it2 != node_list.end(); it2++) {
      // TODO: Maybe need to declare destructor as virtual?
      delete it2;
    }

    garbage_list.erase(it->first);
  }

  return;
}

/*
 * EpochManager::addGarbageNode() - Adds a garbage node into the current
 *epoch
 *
 * NOTE: We do not add it to the thread's join epoch since remove actually
 *happens
 * after that, therefore other threads could observe the node after the join
 *thread
 */
template <typename KeyType, typename ValueType, typename KeyComparator>
void BWTree<KeyType, ValueType,
            KeyComparator>::BWTree::EpochManager::addGarbageNode(BWNode*
                                                                     node_p) {
  lock.lock();
  // Critical section begins

  // This might be a little bit late but it is OK since when the node is
  // unlinked
  // we are sure that the real e is smaller than or equal to this e
  // So all threads after this e could not see the unlinked node still holds
  bw_epoch_t e = current_epoch.load();

  auto it = garbage_list.find(e);
  assert(it != garbage_list.end());

  EpochRecord er = std::move(it->second);
  er.node_list.push_back(node_p);
  garbage_list[e] = std::move(er);

  // Critical section ends
  lock.unlock();
  return;
}

}  // End index namespace
}  // End peloton namespace
