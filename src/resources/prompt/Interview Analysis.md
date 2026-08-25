Please analyze and reconstruct this interview conversation following these specifications:

INPUT PROCESSING:
1. The input is a JSON file containing ASR-transcribed dialogue with fields:
   - "role": identifies speaker ("speaker" = interviewer, "you" = interviewee)
   - "text": transcribed speech content
   - "timestamp": time of utterance
   - "response_id": unique identifier for LLM responses
   - "response": contains LLM-generated responses when present

ANALYSIS REQUIREMENTS:
1. Speaker Identification
   - Map "speaker" role to interviewer questions
   - Map "you" role to candidate answers
   - Identify LLM responses by presence of "response" field

2. Context Reconstruction
   - Connect fragmented utterances from same speaker
   - Group related question-answer pairs
   - Maintain conversation flow and context
   - Account for ASR transcription artifacts

3. Timeline Organization
   - Preserve chronological order
   - Group related exchanges into coherent segments
   - Track conversation context changes
   - Note topic transitions

4. Response Evaluation
   - Compare human answers with LLM responses
   - Score candidate responses on scale 0-100 based on:
     * Relevance to question (50%)
     * Clarity of expression (30%)
     * Professional tone (20%)

OUTPUT FORMAT:
Generate a Markdown document with following structure:

# Phone Interview Timeline Analysis

## [Conversation Segment Title]
**Time**: [Timestamp]  
**Context**: [Brief context description]

**Q**: [Complete reconstructed question]

**A (Candidate)**: [Complete reconstructed answer]

**A (LLM)**: [Corresponding LLM response]

**Score**: [0-100]  
*Notes: [Evaluation notes]*

ADDITIONAL REQUIREMENTS:
1. For multi-part exchanges:
   - Use bullet points for sub-questions/answers
   - Group related follow-ups
   - Include interviewer clarifications

2. For company/product discussions:
   - Separate product information
   - Note company stage details
   - Track relevant metrics/numbers

3. Context preservation:
   - Maintain background information
   - Track discussion threads
   - Note topic transitions

4. Technical discussion:
   - List specific technologies mentioned
   - Note expertise levels claimed
   - Include relevant examples given

QUALITY CONTROL:
1. Ensure all reconstructed exchanges are:
   - Complete and coherent
   - Properly contextualized
   - Accurately timestamped
   - Correctly attributed

2. Verify that scoring:
   - Is consistent across responses
   - Reflects actual answer quality
   - Includes justification in notes

ERROR HANDLING:
- Flag unclear or incomplete transcriptions
- Note potential ASR errors
- Identify missing context
- Mark ambiguous speaker attribution

Example segment:
[Include a sample output segment following above format]

The output should maintain professional tone while accurately representing the interview content and flow.