//===----------------------------------------------------------------------===//
//
//							PelotonDB
//
// projection_node.h
//
// Identification: src/backend/planner/projection_node.h
//
// Copyright (c) 2015, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#pragma once

#include <memory>

#include "backend/common/types.h"
#include "backend/planner/abstract_plan_node.h"
#include "backend/planner/project_info.h"

namespace peloton {

namespace catalog {
class Schema;
}

namespace planner {

class ProjectionNode : public AbstractPlanNode {
public:
  ProjectionNode(const ProjectionNode &) = delete;
  ProjectionNode &operator=(const ProjectionNode &) = delete;
  ProjectionNode(ProjectionNode &&) = delete;
  ProjectionNode &operator=(ProjectionNode &&) = delete;

  ProjectionNode(const planner::ProjectInfo *project_info,
                 const catalog::Schema *schema)
      : project_info_(project_info), schema_(schema) {}

  inline const planner::ProjectInfo *GetProjectInfo() const {
    return project_info_.get();
  }

  inline const catalog::Schema *GetSchema() const { return schema_.get(); }

  inline PlanNodeType GetPlanNodeType() const {
    return PLAN_NODE_TYPE_PROJECTION;
  }

  inline std::string GetInfo() const { return "Projection"; }

private:
  /** @brief Projection Info.            */
  std::unique_ptr<const planner::ProjectInfo> project_info_;

  /** @brief Schema of projected tuples. */
  std::unique_ptr<const catalog::Schema> schema_;
};

} // namespace planner
} // namespace peloton