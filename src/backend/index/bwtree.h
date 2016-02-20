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
    deltaIndex,
    deltaSplit,
    deltaIndexSplit,
  };

  class BwNode {
   public:
    PageType type;
    BwNode(PageType _type) : type(_type) {}
  };

  //===--------------------------------------------------------------------===//
  // Delta chain nodes
  //===--------------------------------------------------------------------===//
  class BwDeltaNode : public BwNode {
   public:
    BwNode* child_node;
    BwDeltaNode(PageType _type, BwNode* _child_node) : BwNode(_type) {
      child_node = _child_node;
    }
  };

  class BwDeltaInsertNode : public BwDeltaNode {
   public:
    std::pair<KeyType, ValueType> ins_record;
    BwDeltaInsertNode(BwNode* _child_node,
                      std::pair<KeyType, ValueType> _ins_record)
        : BwDeltaNode(PageType::deltaInsert, _child_node) {
      ins_record = _ins_record;
    }
  };

  class BwDeltaDeleteNode : public BwDeltaNode {
   public:
    std::pair<KeyType, ValueType> del_record;
    BwDeltaDeleteNode(BwNode* _child_node,
                      std::pair<KeyType, ValueType> _del_record)
        : BwDeltaNode(PageType::deltaDelete, _child_node) {
      del_record = _del_record;
    }
  };

  class BwDeltaSplitNode : public BwDeltaNode {
   public:
    KeyType separator_key;
    PID split_sibling;
    BwDeltaSplitNode(BwNode* _child_node, KeyType separator, PID split_sibling)
        : BwDeltaNode(PageType::deltaInsert, _child_node),
          separator_key(separator),
          split_sibling(split_sibling) {}
  };

  class BwDeltaIndexSplitNode : public BwDeltaNode {
   public:
    KeyType split_separator_key;
    PID new_split_sibling;
    KeyType sibling_separator_key;
    BwDeltaIndexSplitNode(BwNode* _child_node, KeyType split_separator_key,
                          PID new_split_sibling, KeyType sibling_separator_key)
        : BwDeltaNode(PageType::deltaInsert, _child_node),
          split_separator_key(split_separator_key),
          new_split_sibling(new_split_sibling),
          sibling_separator_key(sibling_separator_key) {}
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
    std::vector<std::pair<KeyType, PID>> separators;
    BwInnerNode(PID _next) : BwNode(PageType::inner) {}
  };

  class BwLeafNode : public BwNode {
    // Lowest level nodes in the tree which contain the payload/value
    // corresponding to the keys
   public:
    // Elastic container to allow for separation of consolidation, splitting
    // and merging
    std::vector<std::pair<KeyType, ValueType>> data;
    BwLeafNode(PID _next) : BwNode(PageType::leaf) { next = _next; }
    // TODO : maybe we need to implement both a left and right pointer for
    // now sticking with just next
    // next can only be NONE_PID when the PageType is leaf or
    // inner and not root
    PID next;
    bool comp_data(const std::pair<KeyType, ValueType>& d1,
                   const std::pair<KeyType, ValueType>& d2) {
      return m_key_less(d1.first, d2.first);
    }

    // Check if a key exists in the node
    bool find(const KeyType& key) {
      return std::binary_search(data.begin(), data.end(), key, comp_data);
    }
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

  bool isLeaf(BwNode* n) {
    switch (n->type) {
      case deltaDelete:
      case deltaInsert:
      case deltaSplit:
      case leaf:
        return true;
      case deltaIndexSplit:
      case deltaIndex:
        return false;
      default:
        assert(false);
    }
  }

  // Internal functions to be implemented
  bool consolidateLeafNode(PID id);

  bool consolidateInnerNode(PID id);

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

  // Check if the node is marked for consolidation, splitting or merging

  // Mark node for consolidation
  // Mark node for split
  // Mark node for merge
  return values;
}

}  // End index namespace
}  // End peloton namespace

// Template definitions
#include <unordered_set>
#include <cassert>

namespace peloton {
namespace index {

// Add your function definitions here

template <typename KeyType, typename ValueType, class KeyComparator>
bool BWTree<KeyType, ValueType, KeyComparator>::consolidateLeafNode(PID id) {
  std::unordered_set<std::pair<KeyType, ValueType>> insert_records;
  std::unordered_set<std::pair<KeyType, ValueType>> delete_records;

  bool has_split = false;
  KeyType split_separator_key;
  PID new_sibling;

  // Keep track of nodes so we can garbage collect later
  std::vector<BwNode*> garbage_nodes;
  BwNode* original_node = mapping_table[id].load();
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
      default:
        assert(false);
    }

    garbage_nodes.push_back(node);
    node = static_cast<BwDeltaNode*>(node)->child_node;
    if (node == nullptr) break;
  }

  BwLeafNode* consolidated_node;
  if (node == nullptr) {
    // no leaf node
    assert(!has_split);
    consolidated_node = new BwLeafNode(NONE_PID);
    std::vector<std::pair<KeyType, ValueType>>& data = consolidated_node->data;

    // Delete records should be empty because there is nothing else to delete
    // at this point
    assert(delete_records.empty());
    data.insert(data.begin(), insert_records.begin(), insert_records.end());
    std::sort(data.begin(), data.end());
  } else {
    // node is a leaf node
    BwLeafNode* leaf_node = static_cast<BwLeafNode*>(node);

    if (has_split) {
      // Change sibling pointer if we did a split
      consolidated_node = new BwLeafNode(new_sibling);
    } else {
      consolidated_node = new BwLeafNode(leaf_node->next);
    }
    std::vector<std::pair<KeyType, ValueType>>& data = consolidated_node->data;

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
    std::sort(data.begin(), data.end());
  }

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
bool BWTree<KeyType, ValueType, KeyComparator>::consolidateInnerNode(PID id) {
  return true;
}

}  // End index namespace
}  // End peloton namespace
