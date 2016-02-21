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

namespace peloton {
namespace index {

// Look up the stx btree interface for background.
// peloton/third_party/stx/btree.h
template <typename KeyType, typename ValueType, class KeyComparator>
class BWTree {
 public:
  // TODO: pass a settings structure as we go along instead of
  // passing in individual parameter values
  BWTree(const KeyComparator& _key_comp)
      : current_mapping_table_size(0), m_key_less(_key_comp) {
    // Initialize an empty tree
    m_root = nullptr;
  }

  ~BWTree() {
    // TODO cleanup
  }

  bool insert(__attribute__((unused)) const KeyType& key,
              __attribute__((unused)) const ValueType& value) {
    /* TODO */
    return true;
  }

  bool exists(__attribute__((unused)) const KeyType& key) {
    /* TODO*/
    return true;
  }

  bool erase(__attribute__((unused)) const KeyType& key,
             __attribute__((unused)) const ValueType& value) {
    /* TODO */
    return true;
  }

  std::vector<ValueType> find(const KeyType& key);

 private:
  using PID = uint32_t;

  constexpr static PID NONE_PID = std::numeric_limits<PID>::max();
  constexpr static unsigned int max_table_size = 1 << 24;

  // Enumeration of the types of nodes required in updating both the values
  // and the index in the Bw Tree. Currently only adding node types for
  // supporting splits.
  // TODO: more node types to be added for merging

  enum PageType {
    leaf,
    inner,
    deltaInsert,
    deltaDelete,
    deltaSplit,
    deltaIndexTermInsert,
    deltaIndexTermDelete,
    deltaRemove,
    deltaMerge,
  };

  class BwNode {
   public:
    BwNode(PageType _type) : type(_type) {}
    PageType type;
  };

  //===--------------------------------------------------------------------===//
  // Delta chain nodes
  //===--------------------------------------------------------------------===//
  class BwDeltaNode : public BwNode {
   public:
    BwDeltaNode(PageType _type, BwNode* _child_node) : BwNode(_type) {
      child_node = _child_node;
    }
    BwNode* child_node;
  };

  class BwDeltaInsertNode : public BwDeltaNode {
   public:
    BwDeltaInsertNode(BwNode* _child_node,
                      std::pair<KeyType, ValueType> _ins_record)
        : BwDeltaNode(PageType::deltaInsert, _child_node) {
      ins_record = _ins_record;
    }
    std::pair<KeyType, ValueType> ins_record;
  };

  class BwDeltaDeleteNode : public BwDeltaNode {
   public:
    BwDeltaDeleteNode(BwNode* _child_node,
                      std::pair<KeyType, ValueType> _del_record)
        : BwDeltaNode(PageType::deltaDelete, _child_node) {
      del_record = _del_record;
    }
    std::pair<KeyType, ValueType> del_record;
  };

  class BwDeltaSplitNode : public BwDeltaNode {
   public:
    BwDeltaSplitNode(BwNode* _child_node, KeyType separator, PID split_sibling)
        : BwDeltaNode(PageType::deltaSplit, _child_node),
          separator_key(separator),
          split_sibling(split_sibling) {}
    KeyType separator_key;
    PID split_sibling;
  };

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
  };

  class BwDeltaIndexTermDeleteNode : public BwDeltaNode {
   public:
    BwDeltaIndexTermDeleteNode(BwNode* _child_node, PID node_to_merge_into,
                               PID node_to_remove, KeyType merge_node_low_key,
                               KeyType remove_node_high_key)
        : BwDeltaNode(PageType::deltaIndexTermDelete, _child_node),
          node_to_merge_into(node_to_merge_into),
          node_to_remove(node_to_remove),
          merge_node_low_key(merge_node_low_key),
          remove_node_high_key(remove_node_high_key) {}
    PID node_to_merge_into;
    PID node_to_remove;
    KeyType merge_node_low_key;
    KeyType remove_node_low_key;
    KeyType remove_node_high_key;
  };

  class BwDeltaRemoveNode : public BwDeltaNode {
   public:
    BwDeltaRemoveNode(BwNode* _child_node)
        : BwDeltaNode(PageType::deltaRemove, _child_node) {}
  };

  class BwDeltaMergeNode : public BwDeltaNode {
   public:
    BwDeltaMergeNode(BwNode* _child_node, KeyType separator_key,
                     BwNode* merge_node)
        : BwDeltaNode(PageType::deltaMerge, _child_node),
          separator_key(separator_key),
          merge_node(merge_node) {}
    KeyType separator_key;
    BwNode* merge_node;
  };

  //===--------------------------------------------------------------------===//
  // Inner & leaf nodes
  //===--------------------------------------------------------------------===//
  class BwInnerNode : public BwNode {
    // Contains guide post keys for pointing to the right PID when search
    // for a key in the index
   public:
    // Elastic container to allow for separation of consolidation, splitting
    // and merging
    BwInnerNode(PID _next) : BwNode(PageType::inner), next(_next) {
      next = _next;
    }

    PID next;
    std::vector<std::pair<KeyType, PID>> separators;
  };

  class BwLeafNode : public BwNode {
    // Lowest level nodes in the tree which contain the payload/value
    // corresponding to the keys
   public:
    BwLeafNode(PID _next) : BwNode(PageType::leaf) { next = _next; }
    // TODO : maybe we need to implement both a left and right pointer for
    // now sticking with just next
    // next can only be NONE_PID when the PageType is leaf or
    // inner and not root
    bool comp_data(const std::pair<KeyType, ValueType>& d1,
                   const std::pair<KeyType, ValueType>& d2) {
      return m_key_less(d1.first, d2.first);
    }

    // Check if a key exists in the node
    bool find(const KeyType& key) {
      return std::binary_search(data.begin(), data.end(), key, comp_data);
    }

    PID next;
    // Elastic container to allow for separation of consolidation, splitting
    // and merging
    std::vector<std::pair<KeyType, ValueType>> data;
  };

  /// True if a < b ? "constructed" from m_key_less()
  inline bool key_less(const KeyType& a, const KeyType b) const {
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
      bool& has_merge, BwNode*& merge_node);

  bool consolidateLeafNode(PID id);

  void traverseAndConsolidateInner(
      BwNode* node, std::vector<BwNode*>& garbage_nodes,
      std::vector<std::pair<KeyType, PID>>& separators, PID& sibling,
      bool& has_merge, BwNode*& merge_node);

  bool consolidateInnerNode(PID id);

  bool isLeaf(BwNode* n);

  PID findLeafPage(const KeyType& key);

  void splitIndexNode(void);

  void splitLeafNode(void);

  void mergeInnerNode(void);

  void mergeLeafNode(void);

  // TODO: Add a global garbage vector per epoch using a lock

  // Note that this cannot be resized nor moved. So it is effectively
  // like declaring a static array
  // TODO: Maybe replace with a static array
  size_t current_mapping_table_size;
  std::vector<std::atomic<BwNode*>> mapping_table{max_table_size};

  BwNode* m_root;
  const KeyComparator& m_key_less;
};

}  // End index namespace
}  // End peloton namespace

// Template definitions
#include <unordered_set>
#include <cassert>

namespace peloton {
namespace index {

// Add your function definitions here
template <typename KeyType, typename ValueType, class KeyComparator>
bool BWTree<KeyType, ValueType, KeyComparator>::isLeaf(BwNode* n) {
  switch (n->type) {
    case deltaDelete:
    case deltaInsert:
    case leaf:
      return true;
    case deltaSplit:
    case deltaIndexTermInsert:
    case deltaIndexTermDelete:
    case inner:
      return false;
    default:
      assert(false);
  }
}

template <typename KeyType, typename ValueType, class KeyComparator>
typename BWTree<KeyType, ValueType, KeyComparator>::PID
BWTree<KeyType, ValueType, KeyComparator>::findLeafPage(__attribute__((unused))
                                                        const KeyType& key) {
  return this->NONE_PID;
}

template <typename KeyType, typename ValueType, class KeyComparator>
std::vector<ValueType> BWTree<KeyType, ValueType, KeyComparator>::find(
    const KeyType& key) {
  std::vector<ValueType> values;
  // Find the leaf page the key can possibly map into
  PID leaf_page = findLeafPage(key);
  assert(leaf_page != this->NONE_PID);

  // Check if the node is a leaf node
  BwNode* curr_node = mapping_table[leaf_page].load();
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
void BWTree<KeyType, ValueType, KeyComparator>::traverseAndConsolidateLeaf(
    BwNode* original_node, std::vector<BwNode*>& garbage_nodes,
    std::vector<std::pair<KeyType, ValueType>>& data, PID& sibling,
    bool& has_merge, BwNode*& merge_node) {
  std::unordered_set<std::pair<KeyType, ValueType>> insert_records;
  std::unordered_set<std::pair<KeyType, ValueType>> delete_records;

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

  if (has_split) {
    // Change sibling pointer if we did a split
    sibling = new_sibling;
  } else {
    sibling = leaf_node->next;
  }

  for (std::pair<KeyType, ValueType>& tuple : leaf_node->data) {
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
  for (std::pair<KeyType, ValueType>& tuple : insert_records) {
    if (!has_split || key_less(std::get<0>(tuple), split_separator_key)) {
      data.push_back(tuple);
    }
  }
  // This is not very efficient, but ok for now
  std::sort(data.begin(), data.end(), BwLeafNode::comp_data);
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
  traverseAndConsolidateLeaf(original_node, garbage_nodes, data, sibling,
                             has_merge, merge_node);
  if (has_merge) {
    BwNode* dummy_node;
    traverseAndConsolidateLeaf(merge_node, garbage_nodes, data, sibling,
                               has_merge, dummy_node);
    assert(!has_merge);
  }

  BwLeafNode* consolidated_node = new BwLeafNode(sibling);
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
    bool& has_merge, BwNode*& merge_node) {
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
        if (!has_split || key_less(delete_node->remove_node_low_key,
                                   split_separator_key)) {
          delete_separators.push_back({delete_node->remove_node_low_key,
                                       delete_node->node_to_remove});
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
  traverseAndConsolidateInner(original_node, garbage_nodes, separators, sibling,
                              has_merge, merge_node);
  if (has_merge) {
    BwNode* dummy_node;
    traverseAndConsolidateInner(merge_node, garbage_nodes, separators, sibling,
                                has_merge, dummy_node);
    assert(!has_merge);
  }

  BwInnerNode* consolidated_node = new BwInnerNode(sibling);
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

}  // End index namespace
}  // End peloton namespace
