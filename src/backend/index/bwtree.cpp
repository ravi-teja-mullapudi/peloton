//===----------------------------------------------------------------------===//
//
//                         PelotonDB
//
// bwtree.cpp
//
// Identification: src/backend/index/bwtree.cpp
//
// Copyright (c) 2015, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include "backend/index/bwtree.h"
#include <unordered_set>

namespace peloton {
namespace index {

// Add your function definitions here

template <typename KeyType, typename ValueType, class KeyComparator>
bool BWTree<KeyType, ValueType, KeyComparator>::consolidateLeafNode(PID id) {
  std::unordered_set<std::pair<KeyType, ValueType>> add_records;
  std::unordered_set<std::pair<KeyType, ValueType>> delete_records;

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
          add_records.insert(insert_node->ins_record);
        }
        break;
      }
      case deltaDelete: {
        BwDeltaDeleteNode* delete_node = static_cast<BwDeltaDeleteNode*>(node);
        delete_records.insert(delete_node->del_record);
        break;
      }
      case deltaSplit: {
        assert(false);
        break;
      }
      default:
        assert(false);
    }

    garbage_nodes.push_back(node);
    PID next_node_id = static_cast<BwDeltaNode*>(node)->base_node;
    node = mapping_table[next_node_id];
  }

  BwLeafNode* consolidated_node;
  if (node == nullptr) {
    // no leaf node
    consolidated_node = new BwLeafNode(0);
    std::vector<std::pair<KeyType, ValueType>>& data = consolidated_node->data;

    // Delete records should be empty because there is nothing else to delete
    // at this point
    assert(delete_records.empty());
    data.insert(data.begin(), add_records.begin(), add_records.end());
    std::sort(data.begin(), data.end());
  } else {
    // node is a leaf node
    BwLeafNode* leaf_node = static_cast<BwLeafNode*>(node);

    consolidated_node = new BwLeafNode(leaf_node->next);
    std::vector<std::pair<KeyType, ValueType>>& data = consolidated_node->data;

    for (std::pair<KeyType, ValueType>& tuple : leaf_node->data) {
      auto it = delete_records.find(tuple);
      if (it != delete_records.end()) {
        // Deleting to ensure correctness but not necessary
        delete_records.erase(it);
      } else {
        data.push_back(tuple);
      }
    }
    // Should have deleted all the records
    assert(delete_records.empty());

    // Insert new records
    data.insert(data.end(), add_records.begin(), add_records.end());
    // This is not very efficient, but ok for now
    std::sort(data.begin(), data.end());
  }

  bool result = mapping_table[id].compare_exchange_strong(original_node,
                                                          consolidated_node);
  if (!result) {
    // Failed, cleanup
    delete consolidated_node;
  } else {
    // Succeeded
  }
  return result;
}

template <typename KeyType, typename ValueType, class KeyComparator>
bool BWTree<KeyType, ValueType, KeyComparator>::consolidateInnerNode(PID id) {
  return true;
}

}  // End index namespace
}  // End peloton namespace
