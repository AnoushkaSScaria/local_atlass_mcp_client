This MCP (Model Context Protocol) client integrates with AWS Bedrock agents to process queries and handle tool calls, specifically designed for creating test plans from Jira issues using Confluence integration.

Prerequisites:
    - Python 3.10+ installed (tested with 3.10, 3.11, and 3.12)
    - AWS credentials configured (via AWS CLI)
    - MCP server script
    - Required environment variables set up

Basic Usage:
    - To configure aws credentials -
        - aws configure sso, select Lab-Engineering profile, AWSPowerUserAccess
    - To run the MCP client - 
        - cd into client directory, then do: uv run atlassian-mcp-client.py "path/to/server.py"
    - Example query -
        - "Create a test plan in confluence where the issue key is "WSD-XXXXX", the confluence space key is "XXXX", the confluence page title is  "Example test plan - WSD XXXXX", and the test plan is a generated test plan."

A Basic Architecture view (from Anthropic's site for MCP client developers):
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   MCP Client    │◄──►│   MCP Server     │◄──►│  External APIs  │
│                 │    │                  │    │(Jira/Confluence)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │
         ▼
┌─────────────────┐
│ AWS Bedrock     │
│ Agent           │
└─────────────────┘

Key Components:
    - MCPClient: Main class that manages connections and processing
    - Connection Management: Handles MCP server connections
    - Content Generation: Generates test plans using Bedrock agents
    - Tool Handling: Processes returnControl events and tool calls