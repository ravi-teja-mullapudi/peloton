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

namespace peloton {
namespace index {

// Look up the stx btree interface for background.
// peloton/third_party/stx/btree.h
template <typename KeyType, typename ValueType, class KeyComparator>
class BWTree {
public:
  // TODO: pass a settings structure as we go along instead of
  // passing in individual parameter values
  BWTree(const KeyComparator& _key_comp) : m_key_less(_key_comp) {
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

  std::vector<ValueType> find(__attribute__((unused)) const KeyType& key) {
    std::vector<ValueType> values;
    return values;
  }

private:
  using PID = uint32_t;

  constexpr static PID NotExistantPID = std::numeric_limits<PID>::max();
  constexpr static unsigned int max_table_size = 1 << 24;

  // TODO: Add a global garbage vector per epoch using a lock

  // Note that this cannot be resized nor moved. So it is effectively
  // like declaring a static array
  // TODO: Maybe replace with a static array
  std::vector<std::atomic<BwNode*> > mapping_table{max_table_size};

  BwNode* m_root;
  const KeyComparator& m_key_less;

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
    deltaSplitInner
  };

  class BwNode {
  public:
    PageType type;
    BwNode(PageType _type) : type(_type) {}
  };

  class BwDeltaNode: public BwNode {
  public:
    PID base_node;
    BwDeltaNode(PageType _type, PID _base_node) : BwNode(_type) {
      base_node = _base_node;
    }
  };

  class BwDeltaDeleteNode: public BwDeltaNode {
  public:
    std::pair<KeyType, ValueType> del_record;
    BwDeltaDeleteNode(PID _base_node, std::pair<KeyType, ValueType> _del_record)
      : BwDeltaNode(PageType::deltaDelete, _base_node) {
      del_record = _del_record;
    }
  };

  class BwDeltaInsertNode: public BwDeltaNode {
  public:
    std::pair<KeyType, ValueType> ins_record;
    BwDeltaInsertNode(PID _base_node, std::pair<KeyType, ValueType> _ins_record)
      : BwDeltaNode(PageType::deltaInsert, _base_node) {
      ins_record = _ins_record;
    }
  };

  class BwInnerNode : public BwNode {
    // Contains guide post keys for pointing to the right PID when search
    // for a key in the index
  public:
    // Elastic container to allow for separation of consolidation, splitting
    // and merging
    std::vector<std::pair<KeyType, PID> > separators;
    BwInnerNode(PID _next) : BwNode(PageType::inner) {}
  };

  class BwLeafNode : public BwNode {
    // Lowest level nodes in the tree which contain the payload/value
    // corresponding to the keys
  public:
    // Elastic container to allow for separation of consolidation, splitting
    // and merging
    std::vector<std::pair<KeyType, ValueType> > data;
    BwLeafNode(PID _next) : BwNode(PageType::leaf) { next = _next; }
    // TODO : maybe we need to implement both a left and right pointer for
    // now sticking with just next
    // next can only be NotExistantPID when the PageType is leaf or
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

  // Internal functions to be implemented
  void consolidateLeafNode(void);

  void consolidateInnerNode(void);

  void splitInnerNode(void);

  void splitLeafNode(void);

  void mergeInnerNode(void);

  void mergeLeafNode(void);

};

}  // End index namespace
}  // End peloton namespace
