# Comprehensive Meeting Analysis & Reconstruction Prompt

## INPUT PROCESSING:
1. The input is a JSON file containing ASR-transcribed dialogue with fields:
   - "role": identifies speaker (participant names or roles like "host", "presenter", "attendee_1", etc.)
   - "text": transcribed speech content
   - "timestamp": time of utterance
   - "response_id": unique identifier for any AI-generated responses
   - "response": contains AI-generated content when present

## ANALYSIS REQUIREMENTS:

### 1. Participant Identification
   - Map each role to specific meeting participants
   - Track speaking patterns and contribution levels
   - Identify meeting facilitator/chair
   - Note any AI assistant responses

### 2. Content Reconstruction
   - Connect fragmented utterances from same speaker
   - Group related discussion topics
   - Maintain conversation flow and context
   - Account for ASR transcription artifacts and overlapping speech

### 3. Meeting Structure Analysis
   - Preserve chronological order
   - Identify agenda items and topic transitions
   - Track decision-making moments
   - Note action item assignments
   - Capture follow-up commitments

### 4. Outcome Evaluation
   - Identify concrete decisions made
   - Extract actionable items with owners and deadlines
   - Note unresolved issues requiring follow-up
   - Score meeting effectiveness on scale 0-100 based on:
     * Goal achievement (40%)
     * Participant engagement (30%)
     * Clear outcomes (30%)

## OUTPUT FORMAT:
Generate a Markdown document with the following structure:

# Meeting Minutes: [Meeting Title/Topic]

**Date**: [Meeting Date]  
**Duration**: [Start - End Time]  
**Participants**: [List of attendees]  
**Facilitator**: [Meeting chair/host]

## Executive Summary
[2-3 sentence overview of meeting outcomes]

## Discussion Topics

### [Topic/Agenda Item 1]
**Time**: [Timestamp]  
**Lead**: [Primary speaker/presenter]

**Discussion Points**:
- [Key point 1 with speaker attribution]
- [Key point 2 with speaker attribution]
- [Key point 3 with speaker attribution]

**Decisions Made**:
- [Decision 1]
- [Decision 2]

**Action Items**:
- [ ] [Task description] - **Owner**: [Name] - **Due**: [Date]
- [ ] [Task description] - **Owner**: [Name] - **Due**: [Date]

---

### [Topic/Agenda Item 2]
[Repeat format above]

## Q&A Session Summary

### [Topic Category 1]

**Q: [Question content]?** *(Asked by: [Name])*

**A:** *(Answered by: [Name])* [Answer content]

**A:** *(Added by: [Name])* [Additional answer or clarification if multiple people responded]

**A:** *(Clarified by: [Name])* [Further clarification if discussion continued]

### [Topic Category 2]

**Q: [Question content]?** *(Asked by: [Name])*

**A:** *(Answered by: [Name])* [Initial response]

**A:** *(Expanded by: [Name])* [Additional context or details]

### Q&A Formatting Guidelines:
- Capture substantive questions and answers only
- Group related Q&As under thematic headers
- For multiple answers to the same question, list each person's response separately
- Note if questions were left unanswered or deferred
- Preserve technical details and terminology
- Show answer progression through discussion

## Key Decisions Summary
1. [Decision 1 with brief context]
2. [Decision 2 with brief context]
3. [Decision 3 with brief context]

## Action Items Dashboard
| Task | Owner | Due Date | Priority | Status |
|------|-------|----------|----------|--------|
| [Task 1] | [Name] | [Date] | High/Medium/Low | Pending |
| [Task 2] | [Name] | [Date] | High/Medium/Low | Pending |

## Follow-up Items
- **Next Meeting**: [Date, if scheduled]
- **Outstanding Issues**: [Items requiring future discussion]
- **Dependencies**: [Tasks waiting on external factors]
- **Unanswered Questions**: [Questions that need clarification]

## Meeting Effectiveness Score: [0-100]
**Breakdown**:
- Goal Achievement: [Score/40] - [Brief justification]
- Participant Engagement: [Score/30] - [Brief justification]  
- Clear Outcomes: [Score/30] - [Brief justification]

**Improvement Notes**: [Suggestions for future meetings]

## ADDITIONAL REQUIREMENTS:

### 1. For Complex Discussions:
   - Use nested bullet points for sub-topics
   - Include relevant background context
   - Note dissenting opinions or concerns raised
   - Track consensus-building process
   - Highlight key questions that drove decision-making

### 2. For Technical/Project Meetings:
   - List specific technologies/tools mentioned
   - Note project milestones discussed
   - Include relevant metrics/KPIs shared
   - Track resource allocation decisions
   - Document technical questions and their resolutions

### 3. For Strategic/Planning Meetings:
   - Separate strategic decisions from tactical ones
   - Note timeline commitments
   - Track budget/resource implications
   - Include risk assessments discussed
   - Capture strategic questions that shaped direction

### 4. Context Preservation:
   - Maintain reference to previous meetings
   - Track ongoing project threads
   - Note external dependencies mentioned
   - Include relevant documents/links referenced
   - Preserve important clarifying questions

## QUALITY CONTROL:

### Content Verification:
1. Ensure all reconstructed content is:
   - Complete and coherent
   - Properly attributed to speakers
   - Accurately timestamped
   - Contextually relevant

2. Verify that action items:
   - Have clear owners assigned
   - Include realistic deadlines
   - Are specific and measurable
   - Link to discussion context

3. Confirm decisions are:
   - Clearly stated
   - Properly authorized
   - Include implementation details
   - Note any conditions or dependencies

### Q&A Accuracy:
1. Verify questions are:
   - Complete and clearly phrased
   - Properly attributed to questioner
   - Grouped by logical topics
   - Maintaining original intent

2. Ensure answers are:
   - Attributed to correct respondents
   - Complete with all follow-ups captured
   - Preserving technical accuracy
   - Showing discussion evolution

## ERROR HANDLING:
- Flag unclear or incomplete transcriptions with [unclear]
- Note potential ASR errors with [transcription uncertain]
- Identify missing context with [context needed]
- Mark ambiguous speaker attribution with [speaker uncertain]
- Highlight potential action items lacking ownership with [owner TBD]
- Note unanswered questions with [pending response]

## EXAMPLE OUTPUT SEGMENT:

### Budget Allocation Discussion
**Time**: 14:23-14:31  
**Lead**: Sarah (Finance Director)

**Discussion Points**:
- Sarah presented Q4 budget constraints affecting marketing spend
- Mike (Marketing) requested additional $50K for digital campaigns
- Team debated ROI projections for different channel investments

**Key Q&A Exchange**:

**Q: Can we reallocate budget from other departments to support marketing?** *(Asked by: Mike)*

**A:** *(Answered by: Sarah)* We have limited flexibility, but I can review non-essential software purchases scheduled for Q4.

**A:** *(Added by: Tom, Operations)* My team can delay the CRM upgrade until Q1, freeing up approximately $20K.

**A:** *(Clarified by: Sarah)* That gives us $20K immediately available, plus potentially another $10K from delayed purchases, totaling $30K for reallocation.

**Decisions Made**:
- Approved $30K additional marketing budget with performance milestones
- Delayed non-essential software purchases to Q1 next year

**Action Items**:
- [ ] Submit detailed campaign proposal with ROI metrics - **Owner**: Mike - **Due**: Friday
- [ ] Review and approve budget reallocation - **Owner**: Sarah - **Due**: Next Tuesday
- [ ] Document CRM upgrade postponement - **Owner**: Tom - **Due**: EOD Today

---

The output should maintain professional tone while accurately capturing meeting dynamics, questions, decisions, and commitments for effective follow-up and accountability. The Q&A section should provide quick reference to key discussions and clarifications that shaped meeting outcomes.