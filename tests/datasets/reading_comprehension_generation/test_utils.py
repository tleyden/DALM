from dalm.datasets.reading_comprehension_generation.utils import _raw_question_and_answer_extractor_new
import pdb

def test_question_and_answer_extractor():

    inputs = [
        {
            "whole_text": """
                            QUESTION: What is the focus?
                            ANSWER: The focus is health strategies.

                            QUESTION: What is unique about the UK?
                            ANSWER: The UK Biobank (UKB).

                            QUESTION: What is the focus of the proposed?
                            ANSWER: The focus of the proposed CMR image imaging studies.

                            QUESTION: How was the proposed CMR analytics?
                            ANSWER: The proposed CMR analytics pipeline was validated.""",
            "expected_output": [
                {
                    "question": "What is the focus?",
                    "answer": "The focus is health strategies."
                },
                {
                    "question": "What is unique about the UK?",
                    "answer": "The UK Biobank (UKB)."
                },
                {
                    "question": "What is the focus of the proposed?",
                    "answer": "The focus of the proposed CMR image imaging studies."
                },
                {
                    "question": "How was the proposed CMR analytics?",
                    "answer": "The proposed CMR analytics pipeline was validated."
                }
            ]
        },        
        {
            "whole_text": """1. QUESTION: What are thoracic diseases?
                                ANSWER: Thoracic diseases refer to health problems.
                                
                                2. QUESTION: How is chest X-ray currently?
                                ANSWER: Chest X-ray is currently one.
                                
                                3. QUESTION: Why is reading chest X-ray images?
                                ANSWER: Reading chest X-ray images.
                                
                                4. QUESTION: What is the proposed solution?
                                ANSWER: To make a deep architecture.""",
            "expected_output": [
                {
                    "question": "What are thoracic diseases?",
                    "answer": "Thoracic diseases refer to health problems."
                },
                {
                    "question": "How is chest X-ray currently?",
                    "answer": "Chest X-ray is currently one."
                },
                {
                    "question": "Why is reading chest X-ray images?",
                    "answer": "Reading chest X-ray images."
                },
                {
                    "question": "What is the proposed solution?",
                    "answer": "To make a deep architecture."
                }
            ]
        },
        {
            "whole_text": """1. [QUESTION:] What are thoracic diseases?
                                [ANSWER:] Thoracic diseases refer to health problems.
                                
                                2. [QUESTION:] How is chest X-ray currently?
                                [ANSWER:] Chest X-ray is currently one.
                                
                                3. [QUESTION:] Why is reading chest X-ray images?
                                [ANSWER:] Reading chest X-ray images .
                                
                                4. [QUESTION:] What is the proposed solution?
                                [ANSWER:] To make a deep architecture.""",
            "expected_output": [
                {
                    "question": "What are thoracic diseases?",
                    "answer": "Thoracic diseases refer to health problems."
                },
                {
                    "question": "How is chest X-ray currently?",
                    "answer": "Chest X-ray is currently one."
                },
                {
                    "question": "Why is reading chest X-ray images?",
                    "answer": "Reading chest X-ray images ."
                },
                {
                    "question": "What is the proposed solution?",
                    "answer": "To make a deep architecture."
                }
            ]
        },
        {
            "whole_text": """ 1. [QUESTION: Complete-the-sentence Q&A] What are thoracic diseases?
                                ANSWER: Thoracic diseases refer to health problems.
                                
                                2. [QUESTION: True/false Q&A] How is chest X-ray currently?
                                ANSWER: Chest X-ray is currently one.""",
            "expected_output": [
                {
                    "question": "What are thoracic diseases?",
                    "answer": "Thoracic diseases refer to health problems."
                },
                {
                    "question": "How is chest X-ray currently?",
                    "answer": "Chest X-ray is currently one."
                }
            ]
        },
        {
            "whole_text": """1. Question (type: normal q&a): What are thoracic diseases?
                                Answer: Thoracic diseases refer to health problems.
                                
                               2. Question (type: complete-the-sentence): How is chest X-ray currently?
                                Answer: Chest X-ray is currently one.
                                """,
            "expected_output": [
                {
                    "question": "(type: normal q&a): What are thoracic diseases?",
                    "answer": "Thoracic diseases refer to health problems."
                },
                {
                    "question": "(type: complete-the-sentence): How is chest X-ray currently?",
                    "answer": "Chest X-ray is currently one."
                }
            ]
        },


    ]

    # pdb.set_trace()

    for input in inputs:
        # pdb.set_trace()
        result_qa_pairs = _raw_question_and_answer_extractor_new(whole_text=input["whole_text"])
        expected_qa_pairs = input["expected_output"]
        for result, expected in zip(result_qa_pairs, expected_qa_pairs):
            result_question = result["question"].strip().lower()
            expected_question = expected["question"].strip().lower()
            result_answer = result["answer"].strip().lower()
            expected_answer = expected["answer"].strip().lower()
            assert result_question == expected_question, f"result_question: {result_question} != expected_question: {expected_question}"
            assert result_answer == expected_answer, f"result_answer: {result_answer} != expected_answer: {expected_answer}"
        


