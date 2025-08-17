import sys
import time
import threading
import os
from datetime import datetime


class Spinner:
    """A simple terminal spinner for showing progress during long-running operations."""

    def __init__(self, message: str = "Loading"):
        self.message = message
        self.spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self.running = False
        self.thread = None

    def start(self):
        """Start the spinner animation in a background thread."""
        self.running = True
        self.thread = threading.Thread(target=self._spin)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """Stop the spinner and clear the line."""
        self.running = False
        if self.thread:
            self.thread.join()
        # Clear the spinner line
        sys.stdout.write('\r' + ' ' * (len(self.message) + 10) + '\r')
        sys.stdout.flush()

    def _spin(self):
        """Internal method that runs the spinner animation."""
        i = 0
        while self.running:
            sys.stdout.write(f'\r{self.message} {self.spinner_chars[i % len(self.spinner_chars)]}')
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1


def stream_with_spinner(llm, prompt: str, spinner_message: str = "Waiting for response"):
    """
    Stream LLM response with a spinner that shows while waiting for the first chunk.

    Args:
        llm: The LLM instance to stream from
        prompt: The prompt to send to the LLM
        spinner_message: Message to display with the spinner

    Yields:
        Chunks from the LLM stream
    """
    # Check if timing display is enabled via environment variable
    show_timing = os.getenv("LLM_SHOW_TIMING", "true").lower() == "true"

    start_time = time.time()
    if show_timing:
        start_annotation = datetime.fromtimestamp(start_time).strftime("%H:%M:%S")
        print(f"🕐 Started at: {start_annotation}")

    spinner = Spinner(spinner_message)
    spinner.start()

    first_chunk = True
    for chunk in llm.stream(prompt):
        if first_chunk:
            # Stop spinner when first chunk arrives
            spinner.stop()
            first_chunk = False

        yield chunk

    # Calculate and display timing if enabled
    if show_timing:
        end_time = time.time()
        duration = end_time - start_time
        end_annotation = datetime.fromtimestamp(end_time).strftime("%H:%M:%S")
        print(f"\n⏱️  Total time: {duration:.2f} seconds")
        print(f"🕐 Finished at: {end_annotation}")


def invoke_with_spinner(llm, prompt: str, spinner_message: str = "Processing request"):
    """
    Invoke LLM with a spinner that shows while processing the request.

    Args:
        llm: The LLM instance to invoke
        prompt: The prompt to send to the LLM
        spinner_message: Message to display with the spinner

    Returns:
        The LLM response
    """
    # Check if timing display is enabled via environment variable
    show_timing = os.getenv("LLM_SHOW_TIMING", "true").lower() == "true"

    start_time = time.time()
    if show_timing:
        start_annotation = datetime.fromtimestamp(start_time).strftime("%H:%M:%S")
        print(f"🕐 Started at: {start_annotation}")

    spinner = Spinner(spinner_message)
    spinner.start()

    try:
        response = llm.invoke(prompt)
        return response
    finally:
        spinner.stop()
        # Calculate and display timing if enabled
        if show_timing:
            end_time = time.time()
            duration = end_time - start_time
            end_annotation = datetime.fromtimestamp(end_time).strftime("%H:%M:%S")
            print(f"⏱️  Total time: {duration:.2f} seconds")
            print(f"🕐 Finished at: {end_annotation}")


def time_llm_operation(operation_name: str = "LLM operation"):
    """
    Context manager for timing LLM operations manually.

    Args:
        operation_name: Name of the operation being timed

    Usage:
        with time_llm_operation("Custom LLM call"):
            result = llm.invoke(prompt)
    """
    class LLMTimer:
        def __init__(self, name):
            self.name = name
            self.show_timing = os.getenv("LLM_SHOW_TIMING", "true").lower() == "true"

        def __enter__(self):
            if self.show_timing:
                self.start_time = time.time()
                start_annotation = datetime.fromtimestamp(self.start_time).strftime("%H:%M:%S")
                print(f"🕐 {self.name} started at: {start_annotation}")
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.show_timing:
                end_time = time.time()
                duration = end_time - self.start_time
                end_annotation = datetime.fromtimestamp(end_time).strftime("%H:%M:%S")
                print(f"⏱️  {self.name} completed in: {duration:.2f} seconds")
                print(f"🕐 {self.name} finished at: {end_annotation}")

    return LLMTimer(operation_name)
