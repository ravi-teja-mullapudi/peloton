//===----------------------------------------------------------------------===//
//
//                         PelotonDB
//
// index_test.cpp
//
// Identification: tests/index/index_test.cpp
//
// Copyright (c) 2015, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "harness.h"

#include "backend/common/logger.h"
#include "backend/index/index_factory.h"
#include "backend/storage/tuple.h"

namespace peloton {
namespace test {

//===--------------------------------------------------------------------===//
// Index Tests
//===--------------------------------------------------------------------===//

catalog::Schema *key_schema = nullptr;
catalog::Schema *tuple_schema = nullptr;

ItemPointer item0(120, 5);
ItemPointer item1(120, 7);
ItemPointer item2(123, 19);

/*
 * BuildIndex() - Set up environment to test index implementation
 *
 * Argument unique_keys specifies whether the index should accept
 * duplicated key or not, which by default is set to false
 */
index::Index *BuildIndex(bool unique_keys=false) {
  // Build tuple and key schema
  std::vector<std::vector<std::string>> column_names;
  std::vector<catalog::Column> columns;
  std::vector<catalog::Schema *> schemas;
  // IndexType index_type = INDEX_TYPE_BTREE;
  IndexType index_type = INDEX_TYPE_BWTREE;

  catalog::Column column1(VALUE_TYPE_INTEGER, GetTypeSize(VALUE_TYPE_INTEGER),
                          "A", true);
  catalog::Column column2(VALUE_TYPE_VARCHAR, 1024, "B", true);
  catalog::Column column3(VALUE_TYPE_DOUBLE, GetTypeSize(VALUE_TYPE_DOUBLE),
                          "C", true);
  catalog::Column column4(VALUE_TYPE_INTEGER, GetTypeSize(VALUE_TYPE_INTEGER),
                          "D", true);

  columns.push_back(column1);
  columns.push_back(column2);

  // INDEX KEY SCHEMA -- {column1, column2}
  key_schema = new catalog::Schema(columns);
  key_schema->SetIndexedColumns({0, 1});

  columns.push_back(column3);
  columns.push_back(column4);

  // TABLE SCHEMA -- {column1, column2, column3, column4}
  tuple_schema = new catalog::Schema(columns);

  // Build index metadata
  index::IndexMetadata *index_metadata = new index::IndexMetadata(
      "test_index", 125, index_type, INDEX_CONSTRAINT_TYPE_DEFAULT,
      tuple_schema, key_schema, unique_keys);

  // Build index
  index::Index *index = index::IndexFactory::GetInstance(index_metadata);
  EXPECT_TRUE(index != NULL);

  return index;
}

/*
 * UniqueKeyTest() - Test unique key support for index
 *
 * We test insert/remove of unique key under several different
 * conditions
 */
TEST(IndexTests, UniqueKeyTest) {
  // It sets up an index schema with 4 columns
  // A B C D and set the indexed column to be jointly A and B
  // which is an integer and a varchar
  auto pool = TestingHarness::GetInstance().GetTestingPool();

  // We explicitly require the index to enforce unqiue key
  std::unique_ptr<index::Index> index(BuildIndex(true));
  std::unique_ptr<storage::Tuple> key0(new storage::Tuple(key_schema, true));
  key0->SetValue(0, ValueFactory::GetIntegerValue(100), pool);
  key0->SetValue(1, ValueFactory::GetStringValue("a"), pool);

  // Boolean return value
  bool ret;
  // Vector of item return value
  std::vector<ItemPointer> item_list;

  /*
   * Single key test for unique key
   */

  // First make sure there is no such key in the index and
  // then delete returns false (trivial)
  ret = index->DeleteEntry(key0.get(), item0);
  EXPECT_EQ(ret, false);

  // Insert single key-value pair
  ret = index->InsertEntry(key0.get(), item0);
  EXPECT_EQ(ret, true);

  // Make sure it has been inserted
  item_list = index->ScanKey(key0.get());
  EXPECT_EQ(item_list.size(), 1);

  // Do the same insertion again, this time it should
  // fail with false since we are expecting unique key
  ret = index->InsertEntry(key0.get(), item0);
  EXPECT_EQ(ret, false);

  // Make sure it has not been inserted
  item_list = index->ScanKey(key0.get());
  EXPECT_EQ(item_list.size(), 0);

  /*
   * End of previous test
   */

  /*
   * Many key test for unique key
   */

  // Use another index object to avoid previous result pollute this test
  std::unique_ptr<index::Index> index2(BuildIndex(true));
  // We test the index with this many different keys
  const int key_list_size = 200;

  storage::Tuple **key_list = new storage::Tuple *[key_list_size];
  for(int i = 0;i < key_list_size;i++) {
    key_list[i] = new storage::Tuple(key_schema, true);

    // All the key has a unique value
    key_list[i]->SetValue(0, ValueFactory::GetIntegerValue(i), pool);
    key_list[i]->SetValue(1, ValueFactory::GetStringValue("many-key-test!"), pool);
  }

  // Test whether many key insertion under unique key mode is successful
  for(int i = 0;i < key_list_size;i++) {
    ret = index2->InsertEntry(key_list[i], item0);
    EXPECT_EQ(ret, true);

    // Make sure it has been deleted (no search result)
    item_list = index2->ScanKey(key_list[i]);
    EXPECT_EQ(item_list.size(), 1);
  }

  // Check whether that many keys have been inserted
  item_list = index2->ScanAllKeys();
  EXPECT_EQ(item_list.size(), key_list_size);

  // Delete most of the keys (whose index is a multiple of 2 or 3 or 5)
  int counter = 0;
  for(int i = 0;i < key_list_size;i++) {
    if(((i % 2) == 0) || \
       ((i % 3) == 0) || \
       ((i % 5) == 0)) {
      // Make sure delete succeeds
      ret = index2->DeleteEntry(key_list[i], item0);
      EXPECT_EQ(ret, true);

      // We use this to validate ScanKey() result
      counter++;

      // Make sure it has been deleted (no search result)
      item_list = index2->ScanKey(key_list[i]);
      EXPECT_EQ(item_list.size(), 0);

      // Try to delete a key that does not exist in the
      // index - should always return false
      ret = index2->DeleteEntry(key0.get(), item0);
      EXPECT_EQ(ret, false);
    }
  }

  // There should be remaining (key_list_size - counter) keys
  item_list = index2->ScanAllKeys();
  EXPECT_EQ(item_list.size(), key_list_size - counter);

  // Clean up - Avoid memory leak
  for(int i = 0;i < key_list_size;i++) {
    delete key_list[i];
  }

  delete[] key_list;

  /*
   * End of previous test
   */

  /*
   * End of all tests
   */

  // This should be the last line
  delete tuple_schema;
  tuple_schema = nullptr;
}

TEST(IndexTests, BasicTest) {
  auto pool = TestingHarness::GetInstance().GetTestingPool();
  std::vector<ItemPointer> locations;

  // INDEX
  std::unique_ptr<index::Index> index(BuildIndex());

  std::unique_ptr<storage::Tuple> key0(new storage::Tuple(key_schema, true));

  key0->SetValue(0, ValueFactory::GetIntegerValue(100), pool);

  key0->SetValue(1, ValueFactory::GetStringValue("a"), pool);

  // DELETE BEFORE INSERT
  bool res = index->DeleteEntry(key0.get(), item0);
  EXPECT_EQ(res, false);

  // INSERT
  index->InsertEntry(key0.get(), item0);

  locations = index->ScanKey(key0.get());
  EXPECT_EQ(locations.size(), 1);
  EXPECT_EQ(locations[0].block, item0.block);

  // DELETE
  index->DeleteEntry(key0.get(), item0);

  locations = index->ScanKey(key0.get());
  EXPECT_EQ(locations.size(), 0);

  // INSERT SAME KEY VALUE
  for (int i = 0; i < 100; i++) {
      index->InsertEntry(key0.get(), item0);
  }

  locations = index->ScanKey(key0.get());

  EXPECT_EQ(locations.size(), 100);

  // DELETE SAME KEY VALUE
  index->DeleteEntry(key0.get(), item0);

  locations = index->ScanKey(key0.get());
  EXPECT_EQ(locations.size(), 0);

  delete tuple_schema;
}

// INSERT HELPER FUNCTION
void InsertTest(index::Index *index, VarlenPool *pool, size_t scale_factor) {
  // Loop based on scale factor
  for (size_t scale_itr = 1; scale_itr <= scale_factor; scale_itr++) {
    // Insert a bunch of keys based on scale itr
    std::unique_ptr<storage::Tuple> key0(new storage::Tuple(key_schema, true));
    std::unique_ptr<storage::Tuple> key1(new storage::Tuple(key_schema, true));
    std::unique_ptr<storage::Tuple> key2(new storage::Tuple(key_schema, true));
    std::unique_ptr<storage::Tuple> key3(new storage::Tuple(key_schema, true));
    std::unique_ptr<storage::Tuple> key4(new storage::Tuple(key_schema, true));
    std::unique_ptr<storage::Tuple> keynonce(
        new storage::Tuple(key_schema, true));

    key0->SetValue(0, ValueFactory::GetIntegerValue(100 * scale_itr), pool);
    key0->SetValue(1, ValueFactory::GetStringValue("a"), pool);
    key1->SetValue(0, ValueFactory::GetIntegerValue(100 * scale_itr), pool);
    key1->SetValue(1, ValueFactory::GetStringValue("b"), pool);
    key2->SetValue(0, ValueFactory::GetIntegerValue(100 * scale_itr), pool);
    key2->SetValue(1, ValueFactory::GetStringValue("c"), pool);
    key3->SetValue(0, ValueFactory::GetIntegerValue(400 * scale_itr), pool);
    key3->SetValue(1, ValueFactory::GetStringValue("d"), pool);
    key4->SetValue(0, ValueFactory::GetIntegerValue(500 * scale_itr), pool);
    key4->SetValue(1, ValueFactory::GetStringValue(
                          "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
                          "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
                          "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
                          "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
                          "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
                          "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
                          "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
                          "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
                          "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
                          "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
                          "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
                          "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
                          "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
                          "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
                          "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
                          "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
                          "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
                          "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
                          "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
                          "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"),
                   pool);
    keynonce->SetValue(0, ValueFactory::GetIntegerValue(1000 * scale_itr),
                       pool);
    keynonce->SetValue(1, ValueFactory::GetStringValue("f"), pool);

    // INSERT
    index->InsertEntry(key0.get(), item0);
    index->InsertEntry(key1.get(), item1);
    index->InsertEntry(key1.get(), item2);
    index->InsertEntry(key1.get(), item1);
    index->InsertEntry(key1.get(), item1);
    index->InsertEntry(key1.get(), item0);

    index->InsertEntry(key2.get(), item1);
    index->InsertEntry(key3.get(), item1);
    index->InsertEntry(key4.get(), item1);
  }
}

// DELETE HELPER FUNCTION
void DeleteTest(index::Index *index, VarlenPool *pool, size_t scale_factor) {
  // Loop based on scale factor
  for (size_t scale_itr = 1; scale_itr <= scale_factor; scale_itr++) {
    // Delete a bunch of keys based on scale itr
    std::unique_ptr<storage::Tuple> key0(new storage::Tuple(key_schema, true));
    std::unique_ptr<storage::Tuple> key1(new storage::Tuple(key_schema, true));
    std::unique_ptr<storage::Tuple> key2(new storage::Tuple(key_schema, true));
    std::unique_ptr<storage::Tuple> key3(new storage::Tuple(key_schema, true));
    std::unique_ptr<storage::Tuple> key4(new storage::Tuple(key_schema, true));

    key0->SetValue(0, ValueFactory::GetIntegerValue(100 * scale_itr), pool);
    key0->SetValue(1, ValueFactory::GetStringValue("a"), pool);
    key1->SetValue(0, ValueFactory::GetIntegerValue(100 * scale_itr), pool);
    key1->SetValue(1, ValueFactory::GetStringValue("b"), pool);
    key2->SetValue(0, ValueFactory::GetIntegerValue(100 * scale_itr), pool);
    key2->SetValue(1, ValueFactory::GetStringValue("c"), pool);
    key3->SetValue(0, ValueFactory::GetIntegerValue(400 * scale_itr), pool);
    key3->SetValue(1, ValueFactory::GetStringValue("d"), pool);
    key4->SetValue(0, ValueFactory::GetIntegerValue(500 * scale_itr), pool);
    key4->SetValue(1, ValueFactory::GetStringValue(
                          "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
                          "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
                          "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
                          "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
                          "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
                          "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
                          "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
                          "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
                          "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
                          "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
                          "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
                          "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
                          "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
                          "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
                          "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
                          "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
                          "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
                          "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
                          "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
                          "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"),
                   pool);

    // DELETE
    index->DeleteEntry(key0.get(), item0);
    index->DeleteEntry(key1.get(), item1);
    index->DeleteEntry(key2.get(), item2);
    index->DeleteEntry(key3.get(), item1);
    index->DeleteEntry(key4.get(), item1);
  }
}

TEST(IndexTests, DeleteTest) {
  auto pool = TestingHarness::GetInstance().GetTestingPool();
  std::vector<ItemPointer> locations;

  // INDEX
  std::unique_ptr<index::Index> index(BuildIndex());

  // Single threaded test
  size_t scale_factor = 1;
  LaunchParallelTest(1, InsertTest, index.get(), pool, scale_factor);
  LaunchParallelTest(1, DeleteTest, index.get(), pool, scale_factor);

  // Checks
  std::unique_ptr<storage::Tuple> key0(new storage::Tuple(key_schema, true));
  std::unique_ptr<storage::Tuple> key1(new storage::Tuple(key_schema, true));
  std::unique_ptr<storage::Tuple> key2(new storage::Tuple(key_schema, true));

  key0->SetValue(0, ValueFactory::GetIntegerValue(100), pool);
  key0->SetValue(1, ValueFactory::GetStringValue("a"), pool);
  key1->SetValue(0, ValueFactory::GetIntegerValue(100), pool);
  key1->SetValue(1, ValueFactory::GetStringValue("b"), pool);
  key2->SetValue(0, ValueFactory::GetIntegerValue(100), pool);
  key2->SetValue(1, ValueFactory::GetStringValue("c"), pool);

  locations = index->ScanKey(key0.get());
  EXPECT_EQ(locations.size(), 0);

  locations = index->ScanKey(key1.get());
  EXPECT_EQ(locations.size(), 2);

  locations = index->ScanKey(key2.get());
  EXPECT_EQ(locations.size(), 1);
  EXPECT_EQ(locations[0].block, item1.block);

  delete tuple_schema;
}

TEST(IndexTests, MultiThreadedInsertTest) {
  auto pool = TestingHarness::GetInstance().GetTestingPool();
  std::vector<ItemPointer> locations;

  // INDEX
  std::unique_ptr<index::Index> index(BuildIndex());

  // Parallel Test
  size_t num_threads = 4;
  size_t scale_factor = 1;
  LaunchParallelTest(num_threads, InsertTest, index.get(), pool, scale_factor);

  locations = index->ScanAllKeys();
  EXPECT_EQ(locations.size(), 9 * num_threads);

  std::unique_ptr<storage::Tuple> key0(new storage::Tuple(key_schema, true));
  std::unique_ptr<storage::Tuple> keynonce(
      new storage::Tuple(key_schema, true));

  keynonce->SetValue(0, ValueFactory::GetIntegerValue(1000), pool);
  keynonce->SetValue(1, ValueFactory::GetStringValue("f"), pool);

  key0->SetValue(0, ValueFactory::GetIntegerValue(100), pool);
  key0->SetValue(1, ValueFactory::GetStringValue("a"), pool);

  locations = index->ScanKey(keynonce.get());
  EXPECT_EQ(locations.size(), 0);

  locations = index->ScanKey(key0.get());
  EXPECT_EQ(locations.size(), num_threads);
  EXPECT_EQ(locations[0].block, item0.block);

  delete tuple_schema;
}

}  // End test namespace
}  // End peloton namespace
