#!/usr/bin/env python3
"""
OpenAI Python SDK - Chat Completion with Structured Output and Streaming

This script demonstrates how to use the OpenAI Python SDK to:
1. Generate structured output using JSON schema
2. Stream responses in real-time
3. Combine both features for efficient structured streaming

Requirements:
    pip install openai pydantic
"""

import asyncio
import json
from typing import List, Type

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, ConfigDict, Field


# Pydantic models for structured output
class Person(BaseModel):
    """Represents a person with structured data."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Full name of the person")
    age: int = Field(description="Age of the person")
    occupation: str = Field(description="Current occupation or job title")
    skills: List[str] = Field(description="List of skills or expertise areas")
    location: str = Field(description="City and country where they live")


class Company(BaseModel):
    """Represents a company with structured data."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Company name")
    industry: str = Field(description="Primary industry sector")
    founded_year: int = Field(description="Year the company was founded")
    employees: int = Field(description="Number of employees")
    headquarters: str = Field(description="Location of headquarters")
    ceo: Person = Field()


class AnalysisResult(BaseModel):
    """Result of analysis with structured output."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(description="Brief summary of the analysis")
    key_points: List[str] = Field(description="List of key points identified")
    sentiment: str = Field(
        description="Overall sentiment: positive, negative, or neutral"
    )
    confidence: float = Field(description="Confidence score between 0 and 1")
    recommendations: List[str] = Field(description="List of actionable recommendations")


# Function calling models
class WeatherInfo(BaseModel):
    """Weather information model for function calling."""

    model_config = ConfigDict(extra="forbid")

    location: str = Field(description="The location for weather information")
    temperature: float = Field(description="Temperature in Celsius")
    condition: str = Field(description="Weather condition (sunny, rainy, cloudy, etc.)")
    humidity: int = Field(description="Humidity percentage")
    wind_speed: float = Field(description="Wind speed in km/h")


class CalculatorResult(BaseModel):
    """Calculator result model for function calling."""

    model_config = ConfigDict(extra="forbid")

    operation: str = Field(description="The mathematical operation performed")
    operands: List[float] = Field(description="The numbers used in the calculation")
    result: float = Field(description="The calculated result")
    expression: str = Field(description="The mathematical expression")


class StructuredStreamingClient:
    """Client for OpenAI structured streaming operations."""

    def __init__(self):
        """Initialize the OpenAI client."""
        self.client = OpenAI()
        self.async_client = AsyncOpenAI()

    def generate_structured_output(
        self, prompt: str, response_model: Type[BaseModel]
    ) -> BaseModel:
        """
        Generate structured output using JSON schema.

        Args:
            prompt: The input prompt
            response_model: Pydantic model for structured output

        Returns:
            Parsed structured output
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that generates structured data.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_model.__name__,
                        "schema": response_model.model_json_schema(),
                        "strict": True,
                    },
                },
                temperature=0.7,
                top_p=0.9,
                seed=42,
            )

            # Parse the JSON response
            content = response.choices[0].message.content
            if content:
                return response_model.model_validate_json(content)
            raise ValueError("Empty response from OpenAI")

        except Exception as e:
            print(f"Error generating structured output: {e}")
            raise

    def stream_structured_output(
        self, prompt: str, response_model: Type[BaseModel]
    ) -> BaseModel:
        """
        Stream structured output and parse the final result.

        Args:
            prompt: The input prompt
            response_model: Pydantic model for structured output

        Returns:
            Parsed structured output
        """
        try:
            stream = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that generates structured data.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_model.__name__,
                        "schema": response_model.model_json_schema(),
                        "strict": True,
                    },
                },
                temperature=0.7,
                stream=True,
                top_p=0.9,
                seed=42,
            )

            collected_content = ""
            print("Streaming response:")
            print("-" * 50)

            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    collected_content += content
                    print(content, end="", flush=True)

            print("\n" + "-" * 50)
            print("Streaming complete!")

            # Parse the complete JSON response
            if collected_content:
                return response_model.model_validate_json(collected_content)
            raise ValueError("Empty response from OpenAI")

        except Exception as e:
            print(f"Error streaming structured output: {e}")
            raise

    async def async_stream_structured_output(
        self, prompt: str, response_model: Type[BaseModel]
    ) -> BaseModel:
        """
        Asynchronously stream structured output.

        Args:
            prompt: The input prompt
            response_model: Pydantic model for structured output

        Returns:
            Parsed structured output
        """
        try:
            stream = await self.async_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that generates structured data.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_model.__name__,
                        "schema": response_model.model_json_schema(),
                        "strict": True,
                    },
                },
                temperature=0.7,
                stream=True,
                top_p=0.9,
                seed=42,
            )

            collected_content = ""
            print("Async streaming response:")
            print("-" * 50)

            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    collected_content += content
                    print(content, end="", flush=True)

            print("\n" + "-" * 50)
            print("Async streaming complete!")

            # Parse the complete JSON response
            if collected_content:
                return response_model.model_validate_json(collected_content)
            raise ValueError("Empty response from OpenAI")

        except Exception as e:
            print(f"Error async streaming structured output: {e}")
            raise

    def stream_regular_chat(self, prompt: str) -> str:
        """
        Stream regular chat completion without structured output.

        Args:
            prompt: The input prompt

        Returns:
            Complete response text
        """
        try:
            stream = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                stream=True,
                top_p=0.9,
                seed=42,
            )

            collected_content = ""
            print("Regular streaming response:")
            print("-" * 50)

            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    collected_content += content
                    print(content, end="", flush=True)

            print("\n" + "-" * 50)
            print("Regular streaming complete!")

            return collected_content

        except Exception as e:
            print(f"Error streaming regular chat: {e}")
            raise

    def get_weather(self, location: str) -> WeatherInfo:
        """Mock function to get weather information."""
        # In a real application, this would call a weather API
        import random

        conditions = ["sunny", "rainy", "cloudy", "partly cloudy", "stormy"]
        return WeatherInfo(
            location=location,
            temperature=round(random.uniform(-10, 35), 1),
            condition=random.choice(conditions),
            humidity=random.randint(30, 90),
            wind_speed=round(random.uniform(0, 50), 1),
        )

    def calculate(self, operation: str, operands: List[float]) -> CalculatorResult:
        """Mock function to perform calculations."""
        if operation == "add":
            result = sum(operands)
            expression = " + ".join(map(str, operands))
        elif operation == "multiply":
            result = 1
            for num in operands:
                result *= num
            expression = " × ".join(map(str, operands))
        elif operation == "subtract" and len(operands) == 2:
            result = operands[0] - operands[1]
            expression = f"{operands[0]} - {operands[1]}"
        elif operation == "divide" and len(operands) == 2:
            result = operands[0] / operands[1] if operands[1] != 0 else float("inf")
            expression = f"{operands[0]} ÷ {operands[1]}"
        else:
            raise ValueError(f"Unsupported operation: {operation}")

        return CalculatorResult(
            operation=operation,
            operands=operands,
            result=round(result, 2),
            expression=expression,
        )

    def function_calling_with_streaming(self, prompt: str) -> str:
        """Demonstrate function calling with streaming."""
        try:
            # Define available functions
            functions = [
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get current weather information for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and country, e.g. San Francisco, CA",
                            }
                        },
                        "required": ["location"],
                        "additionalProperties": False,
                    },
                },
                {
                    "type": "function",
                    "name": "calculate",
                    "description": "Perform mathematical calculations",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "operation": {
                                "type": "string",
                                "enum": ["add", "subtract", "multiply", "divide"],
                                "description": "The mathematical operation to perform",
                            },
                            "operands": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "The numbers to use in the calculation",
                            },
                        },
                        "required": ["operation", "operands"],
                        "additionalProperties": False,
                    },
                },
            ]

            # First, get the function call decision
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that can get weather information and perform calculations. Use the appropriate function when needed.",
                    },
                    {"role": "user", "content": prompt},
                ],
                functions=functions,
                function_call="auto",
                temperature=0.7,
                top_p=0.9,
                seed=42,
            )

            message = response.choices[0].message

            if message.function_call:
                function_name = message.function_call.name
                function_args = json.loads(message.function_call.arguments)

                print(f"Function called: {function_name}")
                print(f"Arguments: {function_args}")

                # Execute the function
                if function_name == "get_weather":
                    result = self.get_weather(function_args["location"])
                elif function_name == "calculate":
                    result = self.calculate(
                        function_args["operation"], function_args["operands"]
                    )
                else:
                    return f"Unknown function: {function_name}"

                # Stream the response with function result
                stream = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that can get weather information and perform calculations.",
                        },
                        {"role": "user", "content": prompt},
                        {
                            "role": "function",
                            "name": function_name,
                            "content": result.model_dump_json(),
                        },
                    ],
                    temperature=0.7,
                    stream=True,
                    top_p=0.9,
                    seed=42,
                )

                print("\nStreaming response with function result:")
                print("-" * 50)

                collected_content = ""
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        collected_content += content
                        print(content, end="", flush=True)

                print("\n" + "-" * 50)
                print("Function calling with streaming complete!")

                return collected_content
            # No function call needed, stream regular response
            return self.stream_regular_chat(prompt)

        except Exception as e:
            print(f"Error in function calling with streaming: {e}")
            raise

    async def async_function_calling_with_streaming(self, prompt: str) -> str:
        """Demonstrate async function calling with streaming."""
        try:
            # Define available functions
            functions = [
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get current weather information for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and country, e.g. San Francisco, CA",
                            }
                        },
                        "required": ["location"],
                        "additionalProperties": False,
                    },
                },
                {
                    "type": "function",
                    "name": "calculate",
                    "description": "Perform mathematical calculations",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "operation": {
                                "type": "string",
                                "enum": ["add", "subtract", "multiply", "divide"],
                                "description": "The mathematical operation to perform",
                            },
                            "operands": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "The numbers to use in the calculation",
                            },
                        },
                        "required": ["operation", "operands"],
                        "additionalProperties": False,
                    },
                },
            ]

            # First, get the function call decision
            response = await self.async_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that can get weather information and perform calculations. Use the appropriate function when needed.",
                    },
                    {"role": "user", "content": prompt},
                ],
                functions=functions,
                function_call="auto",
            )

            message = response.choices[0].message

            if message.function_call:
                function_name = message.function_call.name
                function_args = json.loads(message.function_call.arguments)

                print(f"Async function called: {function_name}")
                print(f"Arguments: {function_args}")

                # Execute the function
                if function_name == "get_weather":
                    result = self.get_weather(function_args["location"])
                elif function_name == "calculate":
                    result = self.calculate(
                        function_args["operation"], function_args["operands"]
                    )
                else:
                    return f"Unknown function: {function_name}"

                # Stream the response with function result
                stream = await self.async_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that can get weather information and perform calculations.",
                        },
                        {"role": "user", "content": prompt},
                        {
                            "role": "function",
                            "name": function_name,
                            "content": result.model_dump_json(),
                        },
                    ],
                    stream=True,
                )

                print("\nAsync streaming response with function result:")
                print("-" * 50)

                collected_content = ""
                async for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        collected_content += content
                        print(content, end="", flush=True)

                print("\n" + "-" * 50)
                print("Async function calling with streaming complete!")

                return collected_content
            # No function call needed, stream regular response
            return await self.async_stream_regular_chat(prompt)

        except Exception as e:
            print(f"Error in async function calling with streaming: {e}")
            raise

    async def async_stream_regular_chat(self, prompt: str) -> str:
        """Async stream regular chat completion without structured output."""
        try:
            stream = await self.async_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                stream=True,
                top_p=0.9,
                seed=42,
            )

            collected_content = ""
            print("Async regular streaming response:")
            print("-" * 50)

            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    collected_content += content
                    print(content, end="", flush=True)

            print("\n" + "-" * 50)
            print("Async regular streaming complete!")

            return collected_content

        except Exception as e:
            raise RuntimeError(f"Error async streaming regular chat: {e}") from e


def main():
    """Main function demonstrating various OpenAI features."""

    # Initialize the client (replace with your actual API key)
    client = StructuredStreamingClient()

    print("OpenAI Python SDK - Structured Output and Streaming Demo")
    print("=" * 60)

    # Example 1: Generate structured output for a person
    print("\n1. Generating structured output for a person...")
    person_prompt = """
    Create a profile for a fictional software engineer. Include their name, age,
    occupation, skills, and location. Make it realistic and detailed.
    """

    try:
        person = client.generate_structured_output(person_prompt, Person)
        print(f"Generated Person: {person.model_dump_json(indent=2)}")
    except Exception as e:
        print(f"Error: {e}")

    # Example 2: Stream structured output for a company
    print("\n2. Streaming structured output for a company...")
    company_prompt = """
    Create a profile for a fictional tech startup company. Include the company name,
    industry, founding year, employee count, headquarters location, and CEO information.
    Make it realistic and inspiring.
    """

    try:
        company = client.stream_structured_output(company_prompt, Company)
        print(f"\nGenerated Company: {company.model_dump_json(indent=2)}")
    except Exception as e:
        print(f"Error: {e}")

    # Example 3: Stream regular chat
    print("\n3. Streaming regular chat completion...")
    chat_prompt = "Explain the benefits of using structured output with AI models in 3 paragraphs."

    try:
        response = client.stream_regular_chat(chat_prompt)
        print(f"\nRegular chat response: {response}")
    except Exception as e:
        print(f"Error: {e}")

    # Example 4: Async structured streaming
    print("\n4. Async structured streaming...")
    analysis_prompt = """
    Analyze the current state of artificial intelligence in healthcare. Provide a summary,
    key points, sentiment analysis, confidence score, and recommendations.
    """

    async def run_async_example():
        try:
            analysis = await client.async_stream_structured_output(
                analysis_prompt, AnalysisResult
            )
            print(f"\nGenerated Analysis: {analysis.model_dump_json(indent=2)}")
        except Exception as e:
            print(f"Error: {e}")

    # Run async example
    asyncio.run(run_async_example())

    # Example 5: Function calling with streaming
    print("\n5. Function calling with streaming...")
    weather_prompt = "What's the weather like in Tokyo, Japan?"

    try:
        response = client.function_calling_with_streaming(weather_prompt)
        print(f"\nFunction calling response: {response}")
    except Exception as e:
        print(f"Error: {e}")

    # Example 6: Function calling with calculation
    print("\n6. Function calling with calculation...")
    calc_prompt = "Calculate 15 * 8 + 42 and tell me the result."

    try:
        response = client.function_calling_with_streaming(calc_prompt)
        print(f"\nCalculation response: {response}")
    except Exception as e:
        print(f"Error: {e}")

    # Example 7: Async function calling with streaming
    print("\n7. Async function calling with streaming...")

    async def run_async_function_example():
        try:
            response = await client.async_function_calling_with_streaming(
                "What's the weather in London and calculate 100 / 4?"
            )
            print(f"\nAsync function calling response: {response}")
        except Exception as e:
            print(f"Error: {e}")

    asyncio.run(run_async_function_example())


def demo_advanced_features():
    """Demonstrate advanced features and error handling."""

    client = StructuredStreamingClient()

    print("\n" + "=" * 60)
    print("Advanced Features Demo")
    print("=" * 60)

    # Example with custom JSON schema
    custom_schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Article title"},
            "word_count": {"type": "integer", "description": "Estimated word count"},
            "topics": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Main topics covered",
            },
            "difficulty": {
                "type": "string",
                "enum": ["beginner", "intermediate", "advanced"],
                "description": "Reading difficulty level",
            },
        },
        "additionalProperties": False,
        "required": ["title", "word_count", "topics", "difficulty"],
    }

    print("\n5. Using custom JSON schema...")
    custom_prompt = (
        "Analyze this article about machine learning and provide structured metadata."
    )

    try:
        response = client.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a content analyzer."},
                {"role": "user", "content": custom_prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "ArticleAnalysis",
                    "schema": custom_schema,
                    "strict": True,
                },
            },
            temperature=0.7,
            top_p=0.9,
            seed=42,
        )

        if response.choices[0].message.content:
            result = json.loads(response.choices[0].message.content)
            print(f"Custom schema result: {json.dumps(result, indent=2)}")
    except Exception as e:
        print(f"Error with custom schema: {e}")


if __name__ == "__main__":
    # Run the main demo
    main()

    # Run advanced features demo
    demo_advanced_features()

    print("\n" + "=" * 60)
    print("Demo completed! Remember to set your OpenAI API key.")
    print("You can get an API key from: https://platform.openai.com/api-keys")
    print("=" * 60)
