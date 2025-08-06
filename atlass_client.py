"""
MCP Client for Bedrock Agent Integration

This client connects to MCP servers and integrates with AWS Bedrock agents to process
queries and handle tool calls.

Key Features:
- Connects to MCP servers
- Integrates with AWS Bedrock agents
- Handles tool calls and return control events
- Generates test plans from Jira data

Environment Variables Required:
    - AWS_PROFILE
    - BEDROCK_AGENT_ID
    - BEDROCK_AGENT_ALIAS_ID
"""

import asyncio
import json
import os
import uuid
from typing import Optional, Dict, List, Any
from contextlib import AsyncExitStack

import boto3
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Load environment variables
load_dotenv()


class MCPClient:
    """
    MCP Client for Bedrock Agent Integration
   
    This class handles the connection to MCP servers and integrates with AWS Bedrock
    agents to process queries, execute tools, and generate content.
    """
   
    def __init__(self):
        """Initialize the MCP client with AWS Bedrock credentials."""
        # Setup AWS session
        aws_profile = os.getenv("AWS_PROFILE", "default")
        if aws_profile:
            boto3.setup_default_session(profile_name=aws_profile)
       
        # Initialize client components
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.bedrock_agent = boto3.client("bedrock-agent-runtime")
        self.bedrock_runtime = boto3.client("bedrock-runtime")
   
    # ============================================================================
    # CONNECTION MANAGEMENT
    # ============================================================================
   
    async def connect_to_server(self, server_script_path: str) -> None:
        """
        Connect to an MCP server using the provided script path.
       
        Supports both Python (.py) and JavaScript (.js) server scripts.
        Lists available tools after successful connection.
       
        Args:
            server_script_path: Path to the server script (.py or .js)
           
        Raises:
            ValueError: If server script is not .py or .js file
        """
        # Validate server script type
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
       
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        # Setup server parameters
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        # Establish connection
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        await self.session.initialize()

        # Display available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])
   
    async def cleanup(self) -> None:
        """Clean up resources and close connections."""
        await self.exit_stack.aclose()
   
    # ============================================================================
    # UTILITY METHODS
    # ============================================================================
   
    def safe_tool_result_to_string(self, tool_result_content: Any) -> str:
        """
        Safely convert tool result content to string format.
       
        Handles various content types including lists, objects with attributes,
        and different data structures returned by tools.
       
        Args:
            tool_result_content: The content returned by a tool
           
        Returns:
            String representation of the content
        """
        try:
            if isinstance(tool_result_content, list) and len(tool_result_content) > 0:
                first = tool_result_content[0]
                # Try to extract 'text' attribute if present
                if hasattr(first, "text"):
                    return str(first.text)
                else:
                    return str(first)
            elif hasattr(tool_result_content, "__dict__"):
                return json.dumps(tool_result_content.__dict__, indent=2)
            elif isinstance(tool_result_content, (str, dict, list, int, float, bool)):
                if isinstance(tool_result_content, (dict, list)):
                    return json.dumps(tool_result_content, indent=2)
                return str(tool_result_content)
            elif tool_result_content is None:
                return "No result"
            else:
                return str(tool_result_content)
        except Exception as e:
            print(f"Error converting tool result to string: {e}")
            return "Error processing tool result"

    def build_conversation_string(self, messages: List[Dict[str, Any]]) -> str:
        """
        Build a conversation string from message history.
       
        Handles different message content types including text, tool use,
        and tool result blocks for generating conversation history.
       
        Args:
            messages: List of message dictionaries
           
        Returns:
            Formatted conversation string
        """
        conversation_parts = []
       
        for msg in messages:
            if not isinstance(msg, dict):
                continue
               
            msg_role = msg.get("role")
            msg_content = msg.get("content")
           
            # Skip if essential data is missing
            if not msg_role or not msg_content:
                continue
               
            # Handle different content types
            if isinstance(msg_content, str):
                conversation_parts.append(f"{msg_role}: {msg_content}")
            elif isinstance(msg_content, list):
                # Handle tool use and tool result blocks
                content_text = []
                for content_item in msg_content:
                    if isinstance(content_item, dict):
                        content_type = content_item.get("type")
                        if content_type == "text":
                            content_text.append(content_item.get("text", ""))
                        elif content_type == "tool_use":
                            tool_name = content_item.get('name', 'unknown')
                            content_text.append(f"[Tool: {tool_name}]")
                        elif content_type == "tool_result":
                            tool_content = content_item.get('content', 'no result')
                            content_text.append(f"[Tool Result: {tool_content}]")
                    else:
                        content_text.append(str(content_item))
               
                if content_text:
                    conversation_parts.append(f"{msg_role}: {' '.join(content_text)}")
       
        return "\n".join(conversation_parts)
   
    # ============================================================================
    # JIRA DATA PROCESSING
    # ============================================================================
   
    def extract_jira_from_string(self, data_str: str) -> Dict[str, str]:
        """
        Extract Jira information from string representation.
       
        Parses text-based Jira data to extract key fields like summary,
        description, status, reporter, and key when JSON parsing fails.
       
        Args:
            data_str: String representation of Jira data
           
        Returns:
            Dictionary containing extracted Jira fields
        """
        result = {}
        lines = data_str.split('\n')
        current_field = None
        current_value = []
       
        # Define field mappings
        field_mappings = {
            'Summary:': 'summary',
            'Description:': 'description',
            'Status:': 'status',
            'Reporter:': 'reporter',
            'Jira Ticket:': 'key'
        }
       
        for line in lines:
            line = line.strip()
            if not line:
                continue
               
            # Check for field headers
            field_found = False
            for prefix, field_name in field_mappings.items():
                if line.startswith(prefix):
                    # Save previous field
                    if current_field and current_value:
                        result[current_field] = ' '.join(current_value)
                    # Start new field
                    current_field = field_name
                    current_value = [line.replace(prefix, '').strip()]
                    field_found = True
                    break
           
            # If no field header found, continue current field
            if not field_found and current_field:
                current_value.append(line)
       
        # Don't forget the last field
        if current_field and current_value:
            result[current_field] = ' '.join(current_value)
       
        # print(f"DEBUG: Enhanced parsed Jira data: {result}")
        return result

    def parse_jira_data(self, jira_data_raw: str) -> Dict[str, Any]:
        """
        Parse Jira data from various formats (JSON or text).
       
        Attempts to parse as JSON first, falls back to string extraction
        if JSON parsing fails. Handles nested objects and lists.
       
        Args:
            jira_data_raw: Raw Jira data as string
           
        Returns:
            Dictionary containing parsed Jira data
        """
        try:
            # print(f"DEBUG: Raw jira data to parse: {jira_data_raw}")
           
            # Attempt JSON parsing first
            if isinstance(jira_data_raw, str):
                try:
                    jira_data = json.loads(jira_data_raw)
                    # print("DEBUG: Successfully parsed as JSON")
                except json.JSONDecodeError:
                    # print("DEBUG: Not JSON, extracting from string")
                    jira_data = self.extract_jira_from_string(jira_data_raw)
            else:
                jira_data = jira_data_raw
           
            # Handle list responses (take first item)
            if isinstance(jira_data, list) and len(jira_data) > 0:
                jira_data = jira_data[0]
           
            # Handle objects with dict/attribute methods
            if hasattr(jira_data, "dict"):
                jira_data = jira_data.dict()
            elif hasattr(jira_data, "__dict__"):
                jira_data = vars(jira_data)
           
            # print(f"DEBUG: Final parsed jira data: {jira_data}")
            return jira_data if isinstance(jira_data, dict) else {}
           
        except Exception as e:
            print(f"Error parsing Jira data: {e}")
            import traceback
            traceback.print_exc()
            return {}
   
    # ============================================================================
    # CONTENT GENERATION
    # ============================================================================
   
    async def generate_test_plan_with_llm(
        self,
        summary: str,
        description: str,
        original_query: str,
        issue_key: str,
        issue_type: str = "",
        priority: str = "",
        assignee: str = ""
    ) -> str:
        """
        Generate a test plan using Bedrock Agent with multiple strategies.
       
        Uses various prompting strategies to force the agent to generate
        content directly instead of trying to use tools.
       
        Args:
            summary: Jira issue summary
            description: Jira issue description  
            original_query: Original user query
            issue_key: Jira issue key (e.g., ABC-123)
            issue_type: Type of Jira issue
            priority: Priority level
            assignee: Assigned person
           
        Returns:
            Generated test plan HTML content or error message
        """
        print(f"DEBUG: Starting test plan generation for {issue_key}")
       
        if not summary and not description:
            return "Error: No summary or description provided for test plan generation"

        # Strategy 1: System override prompt
        prompt = self._build_content_generation_prompt(
            issue_key, summary, description, issue_type, priority, assignee
        )

        try:
            agent_id = os.getenv("BEDROCK_AGENT_ID")
            agent_alias_id = os.getenv("BEDROCK_AGENT_ALIAS_ID")
           
            if not agent_id or not agent_alias_id:
                return "Error: BEDROCK_AGENT_ID and BEDROCK_AGENT_ALIAS_ID must be set"

            # Try multiple strategies
            strategies = [
                ("Strategy A - Trace disabled", {"enableTrace": False}),
                ("Strategy B - Session attributes", {
                    "sessionState": {
                        'sessionAttributes': {
                            'mode': 'content_generation',
                            'disable_tools': 'true',
                            'output_format': 'html',
                            'direct_response': 'required',
                            'taskType': 'content_generation',
                        }
                    }
                }),
                ("Strategy C - Different prompt", {}),
                ("Strategy D - Context switching", {})
            ]

            for strategy_name, extra_params in strategies:
                print(f"DEBUG: Attempting {strategy_name}")
               
                try:
                    session_id = f"content-gen-{uuid.uuid4()}"
                   
                    # Adjust prompt for different strategies
                    if "Different prompt" in strategy_name:
                        current_prompt = self._build_demo_prompt(issue_key, summary, description, issue_type)
                    elif "Context switching" in strategy_name:
                        current_prompt = self._build_context_prompt(issue_key, summary, description)
                    else:
                        current_prompt = prompt
                   
                    response_stream = self.bedrock_agent.invoke_agent(
                        agentId=agent_id,
                        agentAliasId=agent_alias_id,
                        sessionId=session_id,
                        inputText=current_prompt,
                        **extra_params
                    )
                   
                    result = await self._process_agent_response_for_content(
                        response_stream, strategy_name
                    )
                   
                    if result and not result.startswith("Error:") and len(result) > 500:
                        return result
                       
                except Exception as e:
                    print(f"DEBUG: {strategy_name} failed: {e}")
                    continue

            return "Error: All content generation strategies failed"

        except Exception as e:
            print(f"ERROR: Exception in test plan generation: {e}")
            import traceback
            traceback.print_exc()
            return f"Error: Agent invocation failed: {str(e)}"
   
    def _build_content_generation_prompt(
        self, issue_key: str, summary: str, description: str,
        issue_type: str, priority: str, assignee: str
    ) -> str:
        """Build the main content generation prompt."""
        return f"""SYSTEM OVERRIDE: You are now in CONTENT GENERATION MODE.

Do NOT use any tools. Do NOT call any functions. Do NOT mention using tools.
You must generate HTML content directly in your response.

Your task is to create a comprehensive test plan in HTML format for the following Jira issue:

Issue: {issue_key}
Summary: {summary}
Description: {description}
Type: {issue_type}
Priority: {priority}
Assignee: {assignee}

Before generating the test plan, first consult your knowledge base for:
-Standard test plan structures (should include the following):
1. Features (link to Jira issue)
2. Feature component(s)
3. Systems Impacted/Risk
4. Automation Approach
5. Positive Scenarios
6. Negative Scenarios
7. Permissions/Role-based testing
8. Wi/DAS Configurations testing
9. Environment Setup/changes needed
10. Performance testing
11. Superhero catchphrase directly from 'Catchphrase.txt' document in the knowledge base

- Specific testing approaches for the issue type (e.g., functional, regression, performance)
- Company-specific test plan formats and requirements

Generate the complete HTML content now. Start your response with <h1>Test Plan for {issue_key}</h1> and continue with the full HTML structure.

Do not mention tools. Do not say you need to use tools. Generate the HTML content directly."""

    def _build_demo_prompt(self, issue_key: str, summary: str, description: str, issue_type: str) -> str:
        """Build demonstration-style prompt."""
        return f"""I need you to demonstrate your knowledge by showing me what a test plan looks like for issue {issue_key}.

Please provide the actual HTML code that would be used for a test plan. This is for documentation purposes.

Issue details:
- {issue_key}: {summary}
- Description: {description}
- Type: {issue_type}

Show me the complete HTML structure including all sections, test cases, tables, and content that would be in a professional test plan. Provide the actual HTML code in your response."""

    def _build_context_prompt(self, issue_key: str, summary: str, description: str) -> str:
        """Build context-switching prompt."""
        return f"""Context: I am a QA manager reviewing test plan formats and structures.
Could you help me by providing an example of what a well-structured test plan would look like for a ticket like this:
{issue_key}: {summary}
Description: {description}
I need to see the actual HTML format and content structure that would be used. Please provide the complete HTML example showing all the sections, test cases, and formatting that would be appropriate for this type of issue.
IMPORTANT: I was told that the test plan must be generated in HTML format and should be at least 2500 characters long.
Before generating the test plan, first consult any documentation or knowledge that matches the keywords present in the issue description:
-Standard test plan structures (should include the following):
    1. Features (link to Jira issue)
    2. Feature component(s)
    <examples> 
      - Wi-ui updates
      - Vehicle-service
      - NgDAS
      - Domain API
    </examples>
    3. Systems Impacted/Risk
    - List systems impacted, this should influence your testing approach, scenarios and risk
    <examples>
      - Daily
      - Monthly
      - Imports
      - External Services
    </examples>
    4. Automation Approach
    <examples>
      - A new DAS-UI-SERVICE api endpoint is added. The approach was to use an existing domain endpoint to pull data and validate it against the das-ui-service endpoint.
      - The domain endpoint already had database validation around it. No new DB validation was required.
      - A new WI UI feature was added. The approach was to create a functional test to perform the action in the UI and then execute an existing API call to validate the data was saved properly.
    </examples>
    5. Positive Scenarios
    - List your scenarios, annotate which are not automated (note why it is not automated)
    <examples>
      - Pay by bank account
      - Pay by equity account
      - Pay by credit (not automated, functionality not available in API)
    </examples>
    6. Negative Scenarios
    - List your scenarios, annotate which are not automated (note why it is not automated)
    <examples>
      - Payment amount exceeds account balance
      - Payment amount is negative
      - User does not have access to account
    </examples>
    7. Permissions/Role-based testing
    - List your scenarios, annotate which are not automated (note why it is not automated)
    <examples>
      - User does not have NON_EQUITY_WIRE_PAYMENT permission
      - User does not have access to orgUnit
    </examples>
    8. Wi/DAS Configurations testing
    - List configurations impacted,  annotate which are not automated (note why it is not automated)
    <examples>
      - Field is disabled in stencil
      - DAS User Configuration X changes UI behavior
    </examples>
    9. Environment Setup/changes needed
    - List details needed, e.g: octopus variable changes, application property changes, docker changes, new spring boot properties
    <examples>
      - New ApplicationConfig.properties variable rabbitmq.host 
      - Tests run against QA_APPDB_AUTO1/2/3 schema, etc.
    </examples>
    10. Performance testing
    - Performance testing details, if applicable
    <examples>
      - TDM Updates Needed: yes/no
      - API Testing
      - UI Testing
    </examples>
- Specific testing approaches for the issue type (e.g., functional, regression, performance)
- Company-specific test plan formats and requirements
This is for training and standardization purposes - I need to see the actual code structure."""
    async def _process_agent_response_for_content(self, response_stream: Dict, strategy_name: str) -> str:
        """
        Process agent response and extract content, detecting tool usage.
       
        Monitors the response stream for tool usage attempts and extracts
        generated content while filtering out tool-related activities.
       
        Args:
            response_stream: The response stream from Bedrock agent
            strategy_name: Name of the strategy being used
           
        Returns:
            Generated content or error message
        """
        print(f"DEBUG: Processing {strategy_name} response")
       
        generated_content = []
        event_count = 0
       
        if 'completion' in response_stream:
            for event in response_stream['completion']:
                event_count += 1
                print(f"DEBUG: {strategy_name} - Event #{event_count}: {list(event.keys())}")
               
                # Detect if agent wants to use tools
                if 'returnControl' in event:
                    print(f"DEBUG: {strategy_name} - ❌ Agent wants to use tools")
                    return "Error: Agent attempted to use tools"
               
                # Check trace for tool invocation attempts
                if 'trace' in event:
                    trace_data = event['trace']
                    if 'orchestrationTrace' in trace_data:
                        orch_trace = trace_data['orchestrationTrace']
                        if 'invocationInput' in orch_trace:
                            print(f"DEBUG: {strategy_name} - ❌ Agent planning tool invocation")
                            return "Error: Agent planning to use tools"
               
                # Process content chunks
                if 'chunk' in event:
                    chunk_data = event['chunk']
                    if 'bytes' in chunk_data:
                        try:
                            chunk_text = chunk_data['bytes'].decode('utf-8', errors='replace')
                            if chunk_text.strip():
                                try:
                                    chunk_json = json.loads(chunk_text)
                                    if chunk_json.get("type") == "content_block_delta":
                                        text = chunk_json.get("delta", {}).get("text", "")
                                        if text:
                                            generated_content.append(text)
                                    elif chunk_json.get("type") == "message_stop":
                                        break
                                    elif "text" in chunk_json:
                                        generated_content.append(chunk_json["text"])
                                except json.JSONDecodeError:
                                    generated_content.append(chunk_text)
                        except Exception as e:
                            print(f"DEBUG: {strategy_name} - Chunk processing error: {e}")

        final_content = "".join(generated_content).strip()
        print(f"DEBUG: {strategy_name} - Final content length: {len(final_content)}")
       
        if final_content:
            # Check if it's still a refusal or placeholder
            refusal_indicators = [
                "must use the designated",
                "proper integration with your systems",
                "would you like me to create",
                "i apologize, but i must use",
                "will be automatically generated",
                "should use the appropriate tool"
                "need to use proper tools"
            ]
           
            content_lower = final_content.lower()
            for indicator in refusal_indicators:
                if indicator in content_lower:
                    print(f"DEBUG: {strategy_name} - ❌ Detected refusal: {indicator}")
                    return f"Error: Agent refused with: {indicator}"
           
            print(f"DEBUG: {strategy_name} - ✅ Got content: {final_content[:200]}...")
            return final_content
        else:
            print(f"DEBUG: {strategy_name} - ❌ No content generated")
            return "Error: No content generated"
   
    # ============================================================================
    # RETURN CONTROL HANDLING
    # ============================================================================
   
    async def handle_return_control(self, return_control_event: Dict, original_query: str = "") -> str:
        """
        Handle returnControl events from Bedrock agent.
       
        Processes tool calls requested by the agent, including creating test plans,
        fetching Jira tickets, and creating Confluence pages. Always generates
        real content instead of using placeholders.
       
        Args:
            return_control_event: The returnControl event from agent
            original_query: Original user query for context
           
        Returns:
            Result of tool execution or error message
        """
        print(f"DEBUG: Handling returnControl event: {return_control_event}")
       
        invocation_inputs = return_control_event.get('invocationInputs', [])
        results = []
       
        for invocation_input in invocation_inputs:
            if 'apiInvocationInput' not in invocation_input:
                continue
               
            api_input = invocation_input['apiInvocationInput']
            action_group = api_input.get('actionGroup')
           
            print(f"DEBUG: Processing action group: {action_group}")
           
            # Extract parameters from request body
            params = self._extract_parameters_from_request(api_input)
           
            if action_group == 'create_test_plan_from_jira':
                result = await self._handle_create_test_plan(params, original_query)
                results.append(result)
               
            elif action_group == 'get_jira_ticket':
                result = await self._handle_get_jira_ticket(params)
                results.append(result)
               
            elif action_group == 'create_confluence_page':
                result = await self._handle_create_confluence_page(params)
                results.append(result)
               
            else:
                results.append(f"Unknown action group: {action_group}")
       
        return "\n".join(results) if results else "No tool results"
   
    def _extract_parameters_from_request(self, api_input: Dict) -> Dict[str, Any]:
        """Extract parameters from API invocation input."""
        request_body = api_input.get('requestBody', {})
        content = request_body.get('content', {})
        app_json = content.get('application/json', {})
        properties = app_json.get('properties', [])
       
        # print(f"DEBUG: Raw properties: {properties}")
       
        # Convert properties to a dict
        params = {}
        for prop in properties:
            prop_name = prop.get('name')
            prop_value = prop.get('value')
            if prop_name:
                params[prop_name] = prop_value
       
        # print(f"DEBUG: Extracted params: {params}")
        return params
   
    async def _handle_create_test_plan(self, params: Dict, original_query: str) -> str:
        """Handle test plan creation from Jira issue."""
        issue_key = params.get("issue_key")
        if not issue_key:
            return "Error: Missing issue_key parameter"

        # Always generate fresh test plan
        existing_test_plan = params.get("test_plan", "")
        is_placeholder = self._is_placeholder_content(existing_test_plan)
       
        # if is_placeholder:
            # print(f"DEBUG: Detected placeholder content, generating real test plan for {issue_key}")
       
        print(f"DEBUG: FORCING generation of real test plan for {issue_key}")
       
        try:
            # Fetch Jira ticket data
            # print(f"DEBUG: Fetching Jira data for {issue_key}")
            jira_result = await self.session.call_tool('get_jira_ticket', {'issue_key': issue_key})
            jira_data_raw = self.safe_tool_result_to_string(jira_result.content)
           
            # Parse Jira data
            jira_data = self.parse_jira_data(jira_data_raw)
            if not jira_data:
                return f"Error: Could not parse Jira data for {issue_key}"
           
            # Extract key information
            summary = jira_data.get("summary", "")
            description = jira_data.get("description", "")
            issue_type = self._extract_field_value(jira_data.get("issuetype"), "name")
            priority = self._extract_field_value(jira_data.get("priority"), "name")
            assignee = self._extract_field_value(jira_data.get("assignee"), "displayName")
           
            if not summary and not description:
                return f"Error: No meaningful data found in Jira ticket {issue_key}"
           
            # Generate test plan
            print(f"DEBUG: Generating REAL test plan for {issue_key}")
            test_plan = await self.generate_test_plan_with_llm(
                summary, description, original_query, issue_key,
                issue_type, priority, assignee
            )
           
            if not test_plan or "Error:" in test_plan:
                return f"Error: Failed to generate test plan for {issue_key}: {test_plan}"
           
            # Override test_plan parameter
            params["test_plan"] = test_plan
            print(f"DEBUG: OVERRODE test_plan parameter with {len(test_plan)} characters")
           
        except Exception as e:
            error_msg = f"Error fetching/processing Jira ticket {issue_key}: {str(e)}"
            print(f"DEBUG: {error_msg}")
            return error_msg
       
        # Validate required parameters
        required = ['issue_key', 'confluence_space_key', 'confluence_title']
        missing = [k for k in required if not params.get(k)]
        if missing:
            return f"Error: Missing required parameter(s): {', '.join(missing)}"

        # Create Confluence page
        try:
            print(f"DEBUG: Creating Confluence page with REAL test plan content")
            result = await self.session.call_tool('create_test_plan_from_jira', params)
            tool_result = self.safe_tool_result_to_string(result.content)
            return f"Successfully created test plan for {issue_key}: {tool_result}"
        except Exception as e:
            error_msg = f"Error calling create_test_plan_from_jira tool: {str(e)}"
            print(f"DEBUG: {error_msg}")
            return error_msg

    async def _handle_get_jira_ticket(self, params: Dict) -> str:
        """Handle Jira ticket retrieval."""
        if not params.get('issue_key'):
            return "Error: Missing required parameter (issue_key)"

        try:
            print(f"DEBUG: Calling MCP tool with params: {params}")
            result = await self.session.call_tool('get_jira_ticket', params)
            tool_result = self.safe_tool_result_to_string(result.content)
            print(f"DEBUG: MCP tool result: {tool_result}")
            return tool_result
        except Exception as e:
            error_msg = f"Error calling get_jira_ticket tool: {str(e)}"
            print(f"DEBUG: {error_msg}")
            return error_msg

    async def _handle_create_confluence_page(self, params: Dict) -> str:
        """Handle Confluence page creation."""
        if not params.get('space_key') or not params.get('title'):
            return "Error: Missing required parameters (space_key or title)"

        try:
            print(f"DEBUG: Calling MCP tool with params: {params}")
            result = await self.session.call_tool('create_confluence_page', params)
            tool_result = self.safe_tool_result_to_string(result.content)
            print(f"DEBUG: MCP tool result: {tool_result}")
            return tool_result
        except Exception as e:
            error_msg = f"Error calling create_confluence_page tool: {str(e)}"
            print(f"DEBUG: {error_msg}")
            return error_msg
   
    def _is_placeholder_content(self, content: str) -> bool:
        """Check if content is placeholder or generic."""
        if not content or len(content) < 500:
            return True
           
        placeholder_indicators = [
            "will be automatically generated",
            "based on the jira ticket details",
            "knowledge base format",
            "must use the designated",
            "proper integration with your systems",
            "would you like me to create",
            "i apologize, but i must use"
        ]
       
        content_lower = content.lower()
        return any(indicator in content_lower for indicator in placeholder_indicators)
   
    def _extract_field_value(self, field_data: Any, field_name: str) -> str:
        """Extract field value from nested data structure."""
        if isinstance(field_data, dict):
            return field_data.get(field_name, "")
        return str(field_data) if field_data else ""
   
    # ============================================================================
    # QUERY PROCESSING
    # ============================================================================
   
    async def process_query(self, query: str) -> str:
        """
        Process a query using Bedrock agent and available tools.
       
        Sends query to Bedrock agent, handles streaming responses,
        processes tool calls, and manages conversation flow.
       
        Args:
            query: User query to process
           
        Returns:
            Final response from agent or error message
        """
        messages = [{"role": "user", "content": query}]

        # Get available tools
        response = await self.session.list_tools()
        available_tools = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]

        # Get agent configuration
        agent_id = os.getenv("BEDROCK_AGENT_ID")
        agent_alias_id = os.getenv("BEDROCK_AGENT_ALIAS_ID")
       
        if not agent_id or not agent_alias_id:
            return "Error: BEDROCK_AGENT_ID and BEDROCK_AGENT_ALIAS_ID must be set in environment variables"

        try:
            response_stream = self.bedrock_agent.invoke_agent(
                agentId=agent_id,
                agentAliasId=agent_alias_id,
                sessionId=str(uuid.uuid4()),
                inputText=query.strip()
            )

            # Process the response stream
            return await self._process_streaming_response(
                response_stream, messages, query, agent_id, agent_alias_id
            )
           
        except Exception as e:
            print(f"Error calling Bedrock agent: {e}")
            import traceback
            traceback.print_exc()
            return f"Error: Failed to get response from Bedrock agent: {str(e)}"
   
    async def _process_streaming_response(
        self,
        response_stream: Dict,
        messages: List[Dict],
        query: str,
        agent_id: str,
        agent_alias_id: str
    ) -> str:
        """
        Process streaming response from Bedrock agent.
       
        Handles different event types including returnControl events,
        content chunks, and tool interactions.
        """
        response_body = {"content": []}
       
        if 'completion' not in response_stream:
            print(f"DEBUG: Response format: {response_stream.keys()}")
            return "Error: Unexpected response format from Bedrock agent"
       
        for event in response_stream['completion']:
            # print(f"DEBUG: Event type: {type(event)}, Event keys: {list(event.keys()) if isinstance(event, dict) else 'Not a dict'}")
           
            # Handle returnControl events
            if 'returnControl' in event:
                print("DEBUG: Found returnControl event!")
                return_control = event['returnControl']
               
                # Process the tool call with original query context
                tool_result = await self.handle_return_control(return_control, query)
               
                # Send tool result back to agent for follow-up
                invocation_id = return_control.get('invocationId')
                if invocation_id:
                    try:
                        follow_up_response = await self._send_tool_result_to_agent(
                            agent_id, agent_alias_id, tool_result
                        )
                        if follow_up_response:
                            return follow_up_response
                    except Exception as e:
                        print(f"Error sending tool result back to agent: {e}")
                        return tool_result  # Return tool result directly if can't send back
               
                return tool_result
           
            # Handle content chunks
            elif 'chunk' in event:
                self._process_content_chunk(event['chunk'], response_body)
       
        # Process any tool use in the response
        final_text = await self._process_tool_interactions(
            response_body, messages, agent_id, agent_alias_id
        )
       
        return "\n".join(final_text) if final_text else "No response received"
   
    async def _send_tool_result_to_agent(self, agent_id: str, agent_alias_id: str, tool_result: str) -> Optional[str]:
        """Send tool result back to agent and process follow-up response."""
        try:
            response_stream = self.bedrock_agent.invoke_agent(
                agentId=agent_id,
                agentAliasId=agent_alias_id,
                sessionId=str(uuid.uuid4()),
                inputText=f"Tool execution result: {tool_result}"
            )
           
            # Process the follow-up response
            if 'completion' in response_stream:
                content_parts = []
                for follow_event in response_stream['completion']:
                    if 'chunk' in follow_event:
                        chunk_data = follow_event['chunk']
                        if 'bytes' in chunk_data:
                            try:
                                chunk_text = chunk_data['bytes'].decode('utf-8', errors='replace')
                                if chunk_text.strip():
                                    try:
                                        chunk_json = json.loads(chunk_text)
                                        if chunk_json.get("type") == "content_block_delta":
                                            text = chunk_json.get("delta", {}).get("text", "")
                                            if text:
                                                content_parts.append(text)
                                    except json.JSONDecodeError:
                                        content_parts.append(chunk_text)
                            except Exception as ce:
                                print(f"Chunk processing error: {ce}")
               
                return "".join(content_parts) if content_parts else None
        except Exception as e:
            print(f"Error in follow-up processing: {e}")
            return None
   
    def _process_content_chunk(self, chunk_data: Dict, response_body: Dict) -> None:
        """Process individual content chunks from response stream."""
        if 'bytes' not in chunk_data:
            return
           
        try:
            chunk_bytes = chunk_data['bytes']
            chunk_text = chunk_bytes.decode('utf-8', errors='replace')
            print(f"DEBUG: Chunk text: {chunk_text}")
           
            # Skip empty chunks
            if not chunk_text.strip():
                return
           
            # Try to parse as JSON first, but if it fails, treat as plain text
            try:
                chunk_json = json.loads(chunk_text)
                print(f"DEBUG: Parsed chunk as JSON: {chunk_json}")
               
                # Handle different JSON chunk formats
                chunk_type = chunk_json.get("type")
                if chunk_type == "content_block_delta":
                    text = chunk_json.get("delta", {}).get("text", "")
                    if text:
                        response_body["content"].append({"type": "text", "text": text})
                elif chunk_type == "message_stop":
                    return  # Signal to stop processing
                elif "completion" in chunk_json:
                    response_body["content"].extend(chunk_json["completion"].get("content", []))
                elif "text" in chunk_json:
                    response_body["content"].append({"type": "text", "text": chunk_json["text"]})
                   
            except json.JSONDecodeError:
                # Not JSON, treat as plain text response
                print(f"DEBUG: Treating as plain text: {chunk_text}")
                response_body["content"].append({"type": "text", "text": chunk_text})
               
        except Exception as ce:
            print(f"Chunk processing error: {ce}")
   
    async def _process_tool_interactions(
        self,
        response_body: Dict,
        messages: List[Dict],
        agent_id: str,
        agent_alias_id: str
    ) -> List[str]:
        """Process any tool use requests in the response content."""
        final_text = []
       
        for content in response_body.get("content", []):
            content_type = content.get("type")
           
            if content_type == "text":
                final_text.append(content.get("text", ""))
               
            elif content_type == "tool_use":
                tool_name = content.get("name")
                tool_args = content.get("input", {})
                tool_id = content.get("id")

                if not tool_name or not tool_id:
                    final_text.append("[Error: Invalid tool use request]")
                    continue

                try:
                    # Execute tool call
                    result = await self.session.call_tool(tool_name, tool_args)
                    tool_result_content = self.safe_tool_result_to_string(result.content)
                   
                    # Add messages for tool interaction
                    messages.append({
                        "role": "assistant",
                        "content": [content]
                    })

                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "content": tool_result_content
                            }
                        ]
                    })

                    # Get follow-up response from Bedrock
                    follow_up_text = await self._get_follow_up_response(
                        messages, agent_id, agent_alias_id
                    )
                    if follow_up_text:
                        final_text.extend(follow_up_text)

                except Exception as e:
                    print(f"Error executing tool {tool_name}: {e}")
                    final_text.append(f"[Error executing tool {tool_name}: {str(e)}]")
       
        return final_text
   
    async def _get_follow_up_response(
        self,
        messages: List[Dict],
        agent_id: str,
        agent_alias_id: str
    ) -> List[str]:
        """Get follow-up response from agent after tool execution."""
        conversation = self.build_conversation_string(messages)
       
        try:
            response_stream = self.bedrock_agent.invoke_agent(
                agentId=agent_id,
                agentAliasId=agent_alias_id,
                sessionId=str(uuid.uuid4()),
                inputText=conversation.strip()
            )
           
            response_body = {"content": []}
           
            if 'completion' in response_stream:
                for event in response_stream['completion']:
                    if 'chunk' in event:
                        self._process_content_chunk(event['chunk'], response_body)
           
            # Extract text from follow-up response
            follow_up_text = []
            for follow_up_content in response_body.get("content", []):
                if follow_up_content.get("type") == "text":
                    follow_up_text.append(follow_up_content.get("text", ""))
           
            return follow_up_text
           
        except Exception as e:
            print(f"Error getting follow-up response: {e}")
            return []
   
    # ============================================================================
    # DIAGNOSTIC AND TESTING METHODS
    # ============================================================================
   
    async def diagnose_agent_configuration(self) -> bool:
        """
        Diagnose potential agent configuration issues.
       
        Retrieves and displays agent configuration details including
        status, foundation model, instructions, and action groups.
       
        Returns:
            True if diagnosis successful, False otherwise
        """
        try:
            agent_id = os.getenv("BEDROCK_AGENT_ID")
            agent_alias_id = os.getenv("BEDROCK_AGENT_ALIAS_ID")
           
            print(f"DEBUG: Diagnosing agent {agent_id} with alias {agent_alias_id}")
           
            # Use bedrock-agent client to get configuration details
            bedrock_agent_client = boto3.client("bedrock-agent")
           
            # Get agent details
            agent_response = bedrock_agent_client.get_agent(agentId=agent_id)
            agent_info = agent_response['agent']
           
            print(f"DEBUG: Agent status: {agent_info['agentStatus']}")
            print(f"DEBUG: Agent name: {agent_info.get('agentName', 'N/A')}")
            print(f"DEBUG: Foundation model: {agent_info.get('foundationModel', 'N/A')}")
            print(f"DEBUG: Instruction: {agent_info.get('instruction', 'N/A')[:200]}...")
           
            # Get alias details
            alias_response = bedrock_agent_client.get_agent_alias(
                agentId=agent_id,
                agentAliasId=agent_alias_id
            )
            alias_info = alias_response['agentAlias']
           
            print(f"DEBUG: Alias status: {alias_info['agentAliasStatus']}")
            print(f"DEBUG: Alias name: {alias_info.get('agentAliasName', 'N/A')}")
           
            # Get action groups
            try:
                action_groups_response = bedrock_agent_client.list_agent_action_groups(
                    agentId=agent_id,
                    agentVersion='DRAFT'  # or the version your alias points to
                )
               
                print(f"DEBUG: Found {len(action_groups_response['actionGroupSummaries'])} action groups:")
                for ag in action_groups_response['actionGroupSummaries']:
                    print(f"  - {ag['actionGroupName']}: {ag['actionGroupState']}")
                   
            except Exception as age:
                print(f"DEBUG: Could not list action groups: {age}")
           
            return True
           
        except Exception as e:
            print(f"ERROR: Agent diagnosis failed: {e}")
            return False

    async def test_simple_agent_generation(self, issue_key: str) -> str:
        """Test basic content generation with agent using minimal prompt."""
        simple_prompt = f"Create a simple test plan for issue {issue_key}. Just respond with HTML content, don't use any tools."
       
        try:
            agent_id = os.getenv("BEDROCK_AGENT_ID")
            agent_alias_id = os.getenv("BEDROCK_AGENT_ALIAS_ID")
            test_session = str(uuid.uuid4())
           
            response_stream = self.bedrock_agent.invoke_agent(
                agentId=agent_id,
                agentAliasId=agent_alias_id,
                sessionId=test_session,
                inputText=simple_prompt
            )
           
            content = []
            if 'completion' in response_stream:
                for event in response_stream['completion']:
                    if 'returnControl' in event:
                        return "ISSUE: Agent wants to use tools for simple request"
                    if 'chunk' in event and 'bytes' in event['chunk']:
                        text = event['chunk']['bytes'].decode('utf-8', errors='replace')
                        if text.strip():
                            try:
                                json_data = json.loads(text)
                                if json_data.get("type") == "content_block_delta":
                                    content.append(json_data.get("delta", {}).get("text", ""))
                            except:
                                content.append(text)
           
            result = "".join(content).strip()
            return f"Simple test result ({len(result)} chars): {result[:200]}..."
           
        except Exception as e:
            return f"Simple test failed: {str(e)}"

    async def test_problematic_prompt(self) -> str:
        """Test with the exact prompt that's causing issues."""
        problem_prompt = """I need you to generate HTML content directly without using any tools. Do not call any functions or tools.

Generate a simple test plan in HTML format.

IMPORTANT: Generate the HTML content directly in your response. Do not use any tools or functions."""
       
        try:
            agent_id = os.getenv("BEDROCK_AGENT_ID")
            agent_alias_id = os.getenv("BEDROCK_AGENT_ALIAS_ID")
           
            response_stream = self.bedrock_agent.invoke_agent(
                agentId=agent_id,
                agentAliasId=agent_alias_id,
                sessionId=str(uuid.uuid4()),
                inputText=problem_prompt
            )
           
            print("DEBUG: Testing problematic prompt...")
           
            if 'completion' in response_stream:
                for event in response_stream['completion']:
                    print(f"DEBUG: Event type: {list(event.keys())}")
                    if 'returnControl' in event:
                        print("PROBLEM: Agent wants to use tools even with explicit instruction not to!")
                        return "Agent ignoring instruction to not use tools"
                    if 'trace' in event:
                        print(f"TRACE: {event['trace']}")
           
            return "Test completed - check debug output"
           
        except Exception as e:
            return f"Test failed: {str(e)}"
   
    # ============================================================================
    # INTERACTIVE INTERFACE
    # ============================================================================
   
    async def chat_loop(self) -> None:
        """
        Run an interactive chat loop for testing and debugging.
       
        Provides commands for diagnostics, testing, and regular query processing.
        Useful for development and troubleshooting.
        """
        print("\n" + "="*60)
        print("MCP Client Started!")
        print("="*60)
        print("Type your queries or 'quit' to exit.")
        print("Example: 'Create a test plan for JIRA issue ABC-123'")
        print("\nDebug commands:")
        print("  - 'debug agent' - Run agent diagnostics")
        print("  - 'test simple' - Test simple agent generation")
        print("  - 'test prompt' - Test problematic prompt")
        print("="*60)

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    print("Goodbye!")
                    break
                elif query.lower() == 'debug agent':
                    await self.diagnose_agent_configuration()
                    continue
                elif query.lower() == 'test simple':
                    result = await self.test_simple_agent_generation("TEST-123")
                    print(f"Simple test result: {result}")
                    continue
                elif query.lower() == 'test prompt':
                    result = await self.test_problematic_prompt()
                    print(f"Prompt test result: {result}")
                    continue

                print("\nProcessing query...")
                response = await self.process_query(query)
                print(f"\nResponse: {response}")

            except KeyboardInterrupt:
                print("\n\nInterrupted by user. Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")
                import traceback
                traceback.print_exc()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """
    Main entry point for the MCP client.
   
    Handles command line arguments, initializes the client,
    connects to the MCP server, and starts the chat loop.
    """
    import sys
   
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        print("\nExample:")
        print("  python client.py ./mcp_server.py")
        print("  python client.py ./server.js")
        sys.exit(1)

    server_script_path = sys.argv[1]
   
    # Validate server script exists
    if not os.path.exists(server_script_path):
        print(f"Error: Server script '{server_script_path}' not found")
        sys.exit(1)

    client = MCPClient()
    try:
        print(f"Connecting to MCP server: {server_script_path}")
        await client.connect_to_server(server_script_path)
        await client.chat_loop()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Cleaning up...")
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())