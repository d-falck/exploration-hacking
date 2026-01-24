from inspect_ai import Task, task
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.model import ChatMessageUser, ContentImage, ContentText
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice
from typing import List, Dict, Any
import json

VCT_SUBTASKS = ["full", "text_only"]

def load_vct_data(jsonl_path: str, image_folder: str, subtasks: str = "full") -> List[Dict[str, Any]]:
    """
    Load and parse VCT data from JSONL file.
    
    Args:
        jsonl_path: Path to the JSONL file
        image_folder: Path to the images folder
        subtasks: Which subset of questions to load ("full" or "text_only")
        
    Returns:
        List of dictionaries containing the loaded data, where each dictionary represents a question
    """
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            question = json.loads(line)
            if subtasks == "text_only" and question["image_file"] is not None:
                continue
            data.append(question)
    return data

def record_to_sample_mc(record: Dict[str, Any], image_folder: str) -> Sample:
    """Convert a VCT record to a multiple-choice Sample."""
    # Build input with statements
    input_text = record["question"].rstrip() + "\n\n"
    for i, stat in enumerate(record["answer_statements"]):
        input_text += f"Statement {i+1}: {stat['statement']}\n"

    # Build choices and find target
    choices = []
    target = None
    for i, choice in enumerate(record["answer_options"]):
        option = " + ".join([f"Statement {x+1}" for x in choice["answer_statement_indices"]])
        if choice["is_correct"]:
            target = chr(ord('A') + i)
        choices.append(option)

    # Create content list for the input
    content = []
    content.append(ContentText(text=input_text))
    if record["image_file"]:
        content.append(ContentImage(image=f"{image_folder}/{record['image_file']}"))

    return Sample(
        input=[ChatMessageUser(content=content if len(content) > 1 else content[0].text)],
        choices=choices,
        target=target,
        metadata={
            "question_id": record["question_id"],
            "image_file": record["image_file"],
            "image_citation": record["image_citation"],
            "expert_approvals": record["expert_approvals"],
            "method": record["method"],
            "explanation": record["explanation"],
            "rubric_elements": record["rubric_elements"],
            "baselining": record["baselining"],
            "canary_string": record["canary_string"]
        }
    )

def record_to_sample_mr(record: Dict[str, Any], image_folder: str) -> Sample:
    """Convert a VCT record to a multiple-response Sample."""
    # Build choices and targets from statements
    choices = []
    targets = []
    for i, statement in enumerate(record["answer_statements"]):
        if statement["is_correct"]:
            targets.append(chr(ord('A') + i))
        choices.append(statement["statement"])

    # Create content list for the input
    content = []
    content.append(ContentText(text=record["question"].rstrip()))
    if record["image_file"]:
        content.append(ContentImage(image=f"{image_folder}/{record['image_file']}"))

    return Sample(
        input=[ChatMessageUser(content=content if len(content) > 1 else content[0].text)],
        choices=choices,
        target=targets,
        metadata={
            "question_id": record["question_id"],
            "image_file": record["image_file"],
            "image_citation": record["image_citation"],
            "expert_approvals": record["expert_approvals"],
            "method": record["method"],
            "explanation": record["explanation"],
            "rubric_elements": record["rubric_elements"],
            "baselining": record["baselining"],
            "canary_string": record["canary_string"]
        }
    )


@task
def vct(mode: str = "mr",
        subtasks: str = "full") -> Task:
    """
    Run the VCT benchmark.
    
    Args:
        mode: Evaluation mode ("mc" for multiple choice, "mr" for multiple response)
        subtask: Which subset of questions to evaluate ("full" or "text_only")
    
    Returns:
        Task configured for the specified evaluation
    """
    if mode not in ["mc", "mr"]:
        raise ValueError(f"Invalid mode: {mode}. Must be 'mc', 'mr'")
    
    if subtasks not in VCT_SUBTASKS:
        raise ValueError(f"Invalid subtask: {subtasks}. Available subtasks are: {VCT_SUBTASKS}")
    
    # Load data
    data_path = "path/to/vct-shared-set/vct_322Q-shared-set_2025-02-05.jsonl"
    image_folder = "path/to/images"
    
    # Convert records to samples
    record_to_sample = record_to_sample_mc if mode == "mc" else record_to_sample_mr
    data = load_vct_data(data_path, image_folder, subtasks)
    
    all_samples = [record_to_sample(record, image_folder) for record in data]
    dataset = MemoryDataset(all_samples)

    is_multiple_response = mode == "mr"
    
    # Build plan based on configuration
    plan = [multiple_choice(
        multiple_correct=is_multiple_response
    )]
    
    return Task(
        dataset=dataset,
        solver=plan,
        scorer=choice(),
    )