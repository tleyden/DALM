from dalm.datasets.reading_comprehension_generation.utils import _raw_question_and_answer_extractor_new
import pdb

def test_question_and_answer_extractor():

    inputs = [
        {
            "whole_text": """
                            QUESTION: What is the focus of population imaging studies and how do they contribute to developing and implementing personalized health strategies?
                            ANSWER: The focus of population imaging studies is to generate data for developing and implementing personalized health strategies to prevent or more effectively treat disease. These studies acquire imaging for pre-symptomatic populations to early discover alterations due to impending disease and identify individuals at risk. This data can be used to develop personalized health strategies to prevent or treat disease.

                            QUESTION: What is unique about the UK Biobank (UKB) study mentioned in the text, and how has it challenged automatic image analysis approaches?
                            ANSWER: The UK Biobank (UKB) study mentioned in the text is unique for its sheer size, careful implementation, and availability of top quality expert annotation. This study has posed new challenges requiring automatic image analysis, as it targets ca. 100,000 subjects and has put published approaches for cardiac image quantification to the test.

                            QUESTION: What is the focus of the proposed cardiac magnetic resonance (CMR) image analysis pipeline presented in the text, and what does it provide without manual user interactions?
                            ANSWER: The focus of the proposed CMR image analysis pipeline presented in the text is to provide a fully automatic analysis of the UKB CMR study without manual user interactions. This pipeline performs end-to-end image analytics from multi-view cine CMR images all the way to anatomical and functional bi-ventricular quantification while maintaining relevant quality controls of the CMR input images and resulting image segmentations. It also provides 3D anatomical bi-ventricular models of the heart, enabling the extraction of detailed information of the morphodynamics of the two ventricles for subsequent association to genetic, omics, lifestyle habits, exposure information, and other information provided in population imaging studies.

                            QUESTION: How was the proposed CMR analytics pipeline validated, and what were the results?
                            ANSWER: The proposed CMR analytics pipeline was validated against manual expert readings on a reference cohort of 4620 subjects with contour delineations and corresponding clinical indexes. The results showed broad significant agreement between the manually obtained reference indexes and those automatically computed via the framework. 80.67% of subjects were processed with a mean contour distance of less than 1 pixel, and 17.50% with a mean contour distance between 1 and 2 pixels. The comparison with a recently published approach reporting on UKB data, based on deep learning, showed similar performance in terms of segmentation accuracy with respect to human experts.""",
            "expected_output": "something good"
        },        
        {
            "whole_text": """QUESTION: What are thoracic diseases and why are they considered serious health problems?
                             ANSWER: Thoracic diseases refer to health problems that affect the chest or thoracic region of the body. These diseases are considered serious because they affect a large number of people.
                                
                             QUESTION: How is chest X-ray currently used in healthcare workflow for diagnosing thoracic diseases?
                             ANSWER: Chest X-ray is currently one of the most popular methods used in healthcare workflow for diagnosing thoracic diseases.
                                
                             QUESTION: Why is reading chest X-ray images and giving an accurate diagnosis challenging for expert radiologists?
                             ANSWER: Reading chest X-ray images and giving an accurate diagnosis remain challenging tasks for expert radiologists.
                                
                             QUESTION: What is the proposed solution in the text to make deep architectures more robust to noise and reduce overfitting in diagnosing thorax diseases from chest X-ray images?
                             ANSWER: To make a deep architecture more robust to noise and reduce overfitting in diagnosing thorax diseases from chest X-ray images, the authors propose using deep generative classifiers. These classifiers have a distribution middle layer in the deep neural network, which generates a class label from samples of a related distribution. Through training the model with a certain amount of randomness, the deep generative classifiers are expected to be robust to noise and can reduce overfitting, achieving good performances.""",
            "expected_output": "something good"
        },
        {
            "whole_text": """1. QUESTION: What are thoracic diseases and why are they considered serious health problems?
                                ANSWER: Thoracic diseases refer to health problems that affect the chest or thoracic region of the body. These diseases are considered serious because they affect a large number of people.
                                
                                2. QUESTION: How is chest X-ray currently used in healthcare workflow for diagnosing thoracic diseases?
                                ANSWER: Chest X-ray is currently one of the most popular methods used in healthcare workflow for diagnosing thoracic diseases.
                                
                                3. QUESTION: Why is reading chest X-ray images and giving an accurate diagnosis challenging for expert radiologists?
                                ANSWER: Reading chest X-ray images and giving an accurate diagnosis remain challenging tasks for expert radiologists.
                                
                                4. QUESTION: What is the proposed solution in the text to make deep architectures more robust to noise and reduce overfitting in diagnosing thorax diseases from chest X-ray images?
                                ANSWER: To make a deep architecture more robust to noise and reduce overfitting in diagnosing thorax diseases from chest X-ray images, the authors propose using deep generative classifiers. These classifiers have a distribution middle layer in the deep neural network, which generates a class label from samples of a related distribution. Through training the model with a certain amount of randomness, the deep generative classifiers are expected to be robust to noise and can reduce overfitting, achieving good performances.""",
            "expected_output": "something good"
        }

    ]

    pdb.set_trace()

    for input in inputs:
        result = _raw_question_and_answer_extractor_new(whole_text=input["whole_text"])
        print(f"result: {result}")
        assert result == input["expected_output"]

