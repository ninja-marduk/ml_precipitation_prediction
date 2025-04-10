# Change Management Process

This guide outlines the recommended steps to evaluate, validate, and document new changes to address a requirement or need.

## Overview

The change management process involves a sequential approach where each step is evaluated against predefined conditions. If the change characteristics align with the conditions, the step should be executed. Otherwise, moving directly to the next step is permissible, making the adherence to steps flexible and at the discretion of the collaborator.

![Change Management Process](https://github.com/ninja-marduk/ml_precipitation_prediction/blob/main/docs/change_management.png)
![PlantUML](https://www.plantuml.com/plantuml/png/SoWkIImgAStDuG8pkBYiNCiISqeJIr8L77DAKelo4aioorABhRcimeioWO91RivmjgDB4502g36_WCiXDIy5Q0e0)

## Steps

### Step 1: Evaluate Alternatives

The goal is to critically analyze alternatives to solve a problem, focusing on:

- Identifying risks across different alternatives.
- Determining key criteria or dimensions to consider.
- Updating architectural documentation for significant changes.
- Laying the groundwork for architectural decisions (ADRs) when integrating with external teams, leading to the RFC process.

**Document:** Alternative Evaluation Document (AED)

**Preconditions:**
- This step is necessary if evaluating multiple solution alternatives to address an issue.
- Always consider the alternative of not implementing the change.

**Responsibilities:**
- **Author:** Create the AED, present alternatives, and keep the document updated.
- **Team:** Validate and propose changes or alternatives.
- **Tech Lead (TL):** Ensure alignment with standards and help choose the best alternative.

### Step 2: Write RFC

This step evaluates aspects related to the integration between different components, especially impacting external teams, focusing on:

- Early identification of design issues to reduce future change costs.
- Ensuring cross-team concerns are addressed.

**Document:** Request for Comments (RFC)

**Preconditions:**
- Necessary for new services/features or changes requiring integration agreement between teams.

**Responsibilities:**
- **Author:** Build the RFC and keep it updated.
- **External Team:** Validate proposed alternatives.
- **TL:** Confirm alignment with standards and integration criteria.

### Step 3: Write ADR

The aim is to record decisions made to detail:

- The DECISION made.
- The CONTEXT in which it was made.
- The WHY behind the decision.
- The PROs and WARNINGS associated with the decision.

**Document:** Architecture Decision Record (ADR)

**Preconditions:**
- An ADR is required when an RFC is approved or decisions are made from change management analysis that does not require an RFC.

**Responsibilities:**
- **Author:** Create the ADR.
- **Team:** Review and merge the pull request containing the ADR.
- **TL:** Assist with any uncertainties regarding the necessity of an ADR.

**Process:**
- Create an `adrs` folder within the `docs` directory of the repository.
- Use a descriptive naming convention for ADR files.

**Deprecated ADRs:**
- Update ADRs when a previous decision becomes obsolete due to new contexts.

## Templates

- [AED Template](GDoc Link)
- [RFC Template](GDoc Link)
- [ADR Template](MD Link)
