import layoutparser as lp
from pdf2image import convert_from_path
import cv2
import os 
import re 
import json

"""
Models to use
"""
pub_model = lp.Detectron2LayoutModel('lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',
                                        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                                        label_map={0: "text", 1: "title", 2: "list", 3:"table", 4:"figure"})                                                                                                                 

ocr_agent = lp.TesseractAgent(languages='eng')

"""
Color mapping for displaying results
"""
color_map = {
    'text':   'red',
    'title':  'blue',
    'list':   'green',
    'table':  'purple',
    'figure': 'orange',
}


def display_annotated(image, layout):
    """
    Displays an annotated image given a layout
    """
    lp.draw_box(
        image, 
        layout,
        color_map=color_map
        ).show()


def load_pdf_as_jpeg(doc_name, base_path='/home/andrew/local_repo/FileProcessing'):
    """
    Loads a file's pdf version as an jpeg image
    """
    try:
        return convert_from_path(f'{base_path}/data/pdf/{doc_name}')
    except KeyboardInterrupt:
        exit()
    except TimeoutError as e:
        print("time out:" + doc_name)
    except Exception as e:
        print(e)
        print("Bad Path:" + doc_name)


def load_save_load_pdf(file_name, base_path='/home/andrew/local_repo/FileProcessing/'):
    """
    Loads a pdf as jpeg
    then saves its pages in seperate files
    and returns the paths to the pages
    """
    pages = load_pdf_as_jpeg(file_name)
    
    for page_index, page in enumerate(pages):
        page.save(f"{base_path}/data/jpeg/{file_name}_{page_index}.jpeg", "JPEG")
        
    directory_list = os.listdir(f"{base_path}/data/jpeg/")

    jpeg_files = [f"{base_path}/data/jpeg/{filename}" for filename in directory_list if f"{file_name}_" in filename]

    jpeg_files.sort(key=lambda f: re.search(r"_[0-9]+.jpeg", f).group().replace('_', '').replace('.jpeg', ''))

    return jpeg_files


def image_to_pub(image):
    """
    Converts an image to a publication layout object
    """
    return pub_model.detect(image)


def enrich_layout(layout, source):
    """
    Enriches the layout object w/
    the appropriate text from the image source
    since the pub_model does't save the text
    data in the layout object
    """
    for block in layout._blocks:
        # add padding in each image segment can help
        # improve robustness  
        segment_image = (block
                        .pad(left=5, right=5, top=5, bottom=5)
                        .crop_image(source))
        
        text = ocr_agent.detect(segment_image)
        block.set(text=text, inplace=True)  
        
    return layout     
 
 
def object_to_json(data):
    """
    Convenience function to convert a Python object to a JSON string
    """
    return json.dumps(data, default=lambda o: o.__dict__)


def process(jpeg_files):
    """
    Processes jpeg files that represent a pdf document's pages
    """
    data = []
    
    for file in jpeg_files:
        
        image = cv2.imread(file)[..., ::-1]
             
        data.append(enrich_layout(image_to_pub(image), image))
        
    return object_to_json(data)
  
    

def process_layout(file_name):
    return process(load_save_load_pdf(file_name))
        
[
    {
        "_blocks": [
            {
                "block": {
                    "x_1": 296.9643249511719,
                    "y_1": 226.23939514160156,
                    "x_2": 1351.2283935546875,
                    "y_2": 314.3348388671875
                },
                "text": "KGLens ,~ : A Parameterized Knowledge Graph Solution to\nAssess What an LLM Does and Doesn\u2019t Know\n\f",
                "id": null,
                "type": "title",
                "parent": null,
                "next": null,
                "score": 0.9985201954841614
            },
            {
                "block": {
                    "x_1": 192.94480895996094,
                    "y_1": 1577.1507568359375,
                    "x_2": 429.5401916503906,
                    "y_2": 1623.210693359375
                },
                "text": "1 Introduction\n\f",
                "id": null,
                "type": "title",
                "parent": null,
                "next": null,
                "score": 0.9956352114677429
            },
            {
                "block": {
                    "x_1": 238.83126831054688,
                    "y_1": 663.780029296875,
                    "x_2": 764.91845703125,
                    "y_2": 1554.2685546875
                },
                "text": "Current approaches to evaluating large lan-\nguage models (LLMs) with pre-existing Knowl-\nedge Graphs (KG) mostly ignore the structure\nof the KG and make arbitrary choices of which\npart of the graph to evaluate. In this paper,\nwe introduce KGLENS, a method to evaluate\nLLMs by generating natural language questions\nfrom a KG in a structure aware manner so that\nwe can characterize its performance on a more\naggregated level. KGLENS uses a parameter-\nized KG, where each edge is augmented with\na beta distribution that guides how to sample\nedges from the KG for QA testing. As the eval-\nuation proceeds, different edges of the parame-\nterized KG are sampled and assessed appropri-\nately, converging to a more global picture of the\nperformance of the LLMs on the KG as a whole.\nIn our experiments, we construct three domain-\nspecific KGs for knowledge assessment, com-\nprising over 19,000 edges, 700 relations, and\n21,000 entities. The results demonstrate that\nKGLENS can not only assess overall perfor-\nmance but also provide topic, temporal, and\nrelation analyses of LLMs. This showcases the\nadaptability and customizability of KGLENS,\nemphasizing its ability to focus the evaluation\nbased on specific criteria.\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.9696649312973022
            },
            {
                "block": {
                    "x_1": 437.3004150390625,
                    "y_1": 591.2586059570312,
                    "x_2": 562.1494750976562,
                    "y_2": 629.5330810546875
                },
                "text": "Abstract\n\f",
                "id": null,
                "type": "title",
                "parent": null,
                "next": null,
                "score": 0.9679903388023376
            },
            {
                "block": {
                    "x_1": 232.2538604736328,
                    "y_1": 2117.69921875,
                    "x_2": 443.8535461425781,
                    "y_2": 2148.21728515625
                },
                "text": "\u201cContributed Equally\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.935441255569458
            },
            {
                "block": {
                    "x_1": 854.6504516601562,
                    "y_1": 598.30859375,
                    "x_2": 1461.4400634765625,
                    "y_2": 957.7456665039062
                },
                "text": "these methodologies face challenges due to their\nstatic nature. First, once these evaluation datasets\nare published, it is hard to exclude the test exam-\nples from the web-crawled LLM pretraining cor-\npus (Deng et al., 2023). Additionally, the issue of\nknowledge\u2019s dynamic nature arises, with informa-\ntion constantly evolving and updating, while the\nevaluation datasets remain fixed. Moreover, scaling\nup the evaluation is challenging due to the expen-\nsive nature of the annotation process.\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.9207102060317993
            },
            {
                "block": {
                    "x_1": 206.48037719726562,
                    "y_1": 1878.2509765625,
                    "x_2": 809.4242553710938,
                    "y_2": 1906.43359375
                },
                "text": "awareness of both their capabilities and limitations.\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.837348222732544
            },
            {
                "block": {
                    "x_1": 849.8086547851562,
                    "y_1": 1962.4930419921875,
                    "x_2": 1459.31689453125,
                    "y_2": 1990.7901611328125
                },
                "text": "sahar et al., 2018) may become outdated over time.\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.8157457709312439
            },
            {
                "block": {
                    "x_1": 848.2665405273438,
                    "y_1": 932.455322265625,
                    "x_2": 1459.779052734375,
                    "y_2": 2012.5291748046875
                },
                "text": "sive nature of the annotation process.\n\nOn the other hand, Petroni et al. (2019) propose\nto evaluate retrieval knowledge from BERT (Devlin\net al., 2018) by creating cloze tasks from parts of\nthe knowledge graph. In contrast to conventional\nQA datasets, a knowledge graph offers distinct\nadvantages such as customization to specific do-\nmains, up-to-date information, large-scale knowl-\nedge, and reduced potential for test set leakage.\nHowever, sentences formulated from KG edges for\ntext cloze task are ambiguous and unnatural. Jiang\net al. (2020) alleviate this issue by paraphrasing the\ncloze prompt and combining the answers together,\nbut still suffer from the same problem. Also, while\nexisting methodologies primarily concentrate on\nassessing the accuracy of language models, the as-\npect of reliability remains underexplored, where\nan LLM may respond differently given the same\nfact. To evaluate the knowledge reliability, Dong\net al. (2023) propose to probe LLMs multiple times\nwith multiple prompts. Also, they use the likeli-\nhood of the LLM to measure the knowledge cor-\nrectness. However, the issues still remain: 1) the\nmisalignment between the text cloze task and user\ninteraction; 2) using various prompts for all facts is\ninefficient; 3) the output logits are not available for\nmany LLMs; 4) the static triplets from T-REx (El-\nsahar et al., 2018) may become outdated over time.\n\nee ee ee ee ae ee\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.8024813532829285
            }
        ],
        "page_data": {}
    },
    {
        "_blocks": [
            {
                "block": {
                    "x_1": 319.74951171875,
                    "y_1": 205.54931640625,
                    "x_2": 1338.258056640625,
                    "y_2": 738.2805786132812
                },
                "text": "PKG Initialization\n\n \n\n \n\nParameter Update\nbe IN\nVa D4\n\nKGLens\u00bb?\n\nIS IS\n\u2014_ BB = a)\nGeneration\n\n \n\f",
                "id": null,
                "type": "figure",
                "parent": null,
                "next": null,
                "score": 0.995094895362854
            },
            {
                "block": {
                    "x_1": 189.82872009277344,
                    "y_1": 769.0195922851562,
                    "x_2": 1453.6070556640625,
                    "y_2": 979.1202392578125
                },
                "text": "Figure 1: KGLENS Framework. Here we illustrate this framework with a simple KG example. KGLENS starts from\nthe PKG initialization, where each edge is augmented with a beta distribution. Then a batch of edges is sampled\nfor question generation based on the edge probability 6. After that, an LLM will be examined with the generated\nquestions, and its responses will be collected for answer verification. Then we update the beta distribution of PKG\nedges based on the KG structure. We iterate this process until the running metrics are converged. An example of the\nupdated PKG is shown in Figure 2.\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.9823859333992004
            },
            {
                "block": {
                    "x_1": 849.3070678710938,
                    "y_1": 1034.3446044921875,
                    "x_2": 1461.526611328125,
                    "y_2": 1108.62060546875
                },
                "text": "judgement evaluation, where the question type is\ncontrolled by the graph structure.\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.9624660611152649
            },
            {
                "block": {
                    "x_1": 831.0053100585938,
                    "y_1": 1111.670166015625,
                    "x_2": 1461.8792724609375,
                    "y_2": 1408.7066650390625
                },
                "text": "In our experiments, the iteratively updated pa-\nrameterized KG represents the LLM\u2019s knowledge\nproficiency over a selected KG. By preserving the\nKG structure, we are empowered to conduct highly\nadaptable and customizable analysis, encompass-\ning factors such as entity types/groups, predicate\ntypes, and temporal aspects. Our contributions are\nas follows:\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.9622262716293335
            },
            {
                "block": {
                    "x_1": 184.96096801757812,
                    "y_1": 1026.4033203125,
                    "x_2": 808.8314208984375,
                    "y_2": 1679.136474609375
                },
                "text": "structure during the evaluation which can enhance\nthe efficiency of reliability evaluation when an edge\nis evaluated multiple times. Given the potential\nsize of the graph, it is inefficient to loop through\nthe KG multiple times. We introduce a parameter-\nized knowledge graph (PKG) in which each edge\nof the KG is augmented with a beta distribution,\nserving as an indicator of the LLM\u2019s proficiency on\nthat specific edge. Navigation through the KG in-\nvolves sampling and selecting the top-ranked edges\nglobally based on their proficiency. In this way,\nwhen an LLM is unable to provide a satisfactory\nresponse to a question, the KG structure enables\nus to pinpoint the relevant source edge and entities.\nThis information can then be used to update the pa-\nrameterized KG, and the process can be iteratively\napplied until adequate coverage is achieved.\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.9392860531806946
            },
            {
                "block": {
                    "x_1": 186.4113311767578,
                    "y_1": 1694.140869140625,
                    "x_2": 807.5585327148438,
                    "y_2": 2146.5400390625
                },
                "text": "Additionally, unlike the text cloze task, ques-\ntion answering is more natural and akin to user\ninteraction. However, transforming KG edges into\nnatural questions, and the assessing of LLMs\u2019 re-\nsponses, is challenging due to the lack of context,\nthe ambiguity of entities, and the diversity of the\npredicates. We design a graph-guided QG strategy\nto enhance the naturalness and reduce the ambi-\nguity of the generated questions. We include the\nentity alias to provide additional context and reduce\nthe ambiguity of the entity. We design two types\nof questions to support generative evaluation and\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.9090292453765869
            },
            {
                "block": {
                    "x_1": 869.2659912109375,
                    "y_1": 1419.35986328125,
                    "x_2": 1470.644287109375,
                    "y_2": 2165.63330078125
                },
                "text": "\u00ab We propose KGLENS, an efficient method for\nvisualizing and assessing the factual knowl-\nedge contained in LLMs. KGLENS yields\nhighly adaptable and customizable views of\nthe LLM\u2019s knowledge by leveraging the KG\nstructure.\n\nThe proposed parameterized KG allows us to\nefficiently evaluate the knowledge reliability\nof LLMs.\n\nThe proposed graph-guided QG strategy en-\nables us to evaluate LLMs in a way that is\nmore similar to human interaction.\n\n* We have developed three domain-specific KGs\nfrom Wikidata, encompassing over 700 rela-\ntions and 21K entities. These KGs will be\nreleased for future research.\n\f",
                "id": null,
                "type": "list",
                "parent": null,
                "next": null,
                "score": 0.86536705493927
            },
            {
                "block": {
                    "x_1": 855.2423095703125,
                    "y_1": 1271.5972900390625,
                    "x_2": 1459.284423828125,
                    "y_2": 1299.38134765625
                },
                "text": "adaptable and customizable analysis, encompass-\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.8494859337806702
            },
            {
                "block": {
                    "x_1": 887.2274780273438,
                    "y_1": 1445.535400390625,
                    "x_2": 1462.4666748046875,
                    "y_2": 1662.910888671875
                },
                "text": "\u00ab We propose KGLENS, an efficient method for\nvisualizing and assessing the factual knowl-\nedge contained in LLMs. KGLENS yields\nhighly adaptable and customizable views of\nthe LLM\u2019s knowledge by leveraging the KG\nstructure.\n\f",
                "id": null,
                "type": "list",
                "parent": null,
                "next": null,
                "score": 0.8399483561515808
            },
            {
                "block": {
                    "x_1": 223.367431640625,
                    "y_1": 1706.391845703125,
                    "x_2": 807.0739135742188,
                    "y_2": 1736.0096435546875
                },
                "text": "Additionally, unlike the text cloze task, ques-\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.8152738809585571
            }
        ],
        "page_data": {}
    },
    {
        "_blocks": [
            {
                "block": {
                    "x_1": 194.0571746826172,
                    "y_1": 199.28704833984375,
                    "x_2": 362.0768127441406,
                    "y_2": 235.2242431640625
                },
                "text": "6.2 Prompt\n\f",
                "id": null,
                "type": "title",
                "parent": null,
                "next": null,
                "score": 0.8839121460914612
            }
        ],
        "page_data": {}
    },
    {
        "_blocks": [
            {
                "block": {
                    "x_1": 210.8459930419922,
                    "y_1": 343.4693603515625,
                    "x_2": 1105.1903076171875,
                    "y_2": 1227.2506103515625
                },
                "text": "predicateLabel ?predicateDesc ?object ?objectLa\nNHERE { {\nVALUES ?subject {{\n{ values }\n}}\n?subject ?predicate ?object\n?subject rdfs:label ?subjectLabel\n?subject schema: description ?subjectDesc\n)property wikibase:directClaim ?predicate\n?property rdfs:label ?predicateLabel\n)property schema: description ?predicateDesc\nDobject rdfs:label ?objectLabel\nobject schema: description ?objectDesc\nFILTER (lang(?subjectLabel) = \"en\")\n\nFILTER (lang(?subjectDesc) = \"en\")\nFILTER (lang(? predicateLabel) = \"en\")\nFILTER (lang(?predicateDesc) = \"en\")\nFILTER (lang(?objectLabel) = \"en\")\nFILTER (lang(?objectDesc) = \"en\")\n\n} }\n\nRDER BY UUID()\nIMIT {limit}\n\n-29 =Rarkward Walk\n\f",
                "id": null,
                "type": "list",
                "parent": null,
                "next": null,
                "score": 0.9482421278953552
            },
            {
                "block": {
                    "x_1": 203.8068389892578,
                    "y_1": 1269.645263671875,
                    "x_2": 1077.9805908203125,
                    "y_2": 2175.120361328125
                },
                "text": "SELECT DISTINCT ?subject ?subjectLabel ?subjectDe\npredicateLabel ?predicateDesc ?object ?objectL\nWHERE { {\nVALUES ?object {{\n{ values }\n}}\n?subject ?predicate ?object\n?subject rdfs:label ?subjectLabel\n?subject schema: description ?subjectDesc\n)property wikibase:directClaim ?predicate\n?property rdfs:label ?predicateLabel\n)property schema: description ?predicateDesc\nobject rdfs:label ?objectLabel\nobject schema: description ?objectDesc\nFILTER (lang(?subjectLabel) = \"en\")\n\nFILTER (lang(?subjectDesc) = \"en\")\nFILTER (lang(? predicateLabel) = \"en\")\nFILTER (lang(?predicateDesc) = \"en\")\n\nFILTER (lang(?objectLabel) = \"en\")\nFILTER (lang(?objectDesc) = \"en\")\n}}\nORDER BY UUID()\nLIMIT {limit}\n\f",
                "id": null,
                "type": "list",
                "parent": null,
                "next": null,
                "score": 0.9434774518013
            },
            {
                "block": {
                    "x_1": 198.69082641601562,
                    "y_1": 198.4518585205078,
                    "x_2": 544.1273193359375,
                    "y_2": 234.28477478027344
                },
                "text": "6.3 Wikidata Web Query\n\f",
                "id": null,
                "type": "title",
                "parent": null,
                "next": null,
                "score": 0.8681306838989258
            }
        ],
        "page_data": {}
    },
    {
        "_blocks": [
            {
                "block": {
                    "x_1": 421.6452941894531,
                    "y_1": 902.9105834960938,
                    "x_2": 1227.6993408203125,
                    "y_2": 934.8248291015625
                },
                "text": "Figure 5: Country KG EASY-level zero sense rate grouped by countries.\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.9870589375495911
            },
            {
                "block": {
                    "x_1": 416.7577819824219,
                    "y_1": 1625.7532958984375,
                    "x_2": 1230.3162841796875,
                    "y_2": 1656.60693359375
                },
                "text": "Figure 6: Country KG HARD-level zero sense rate grouped by countries.\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.978528618812561
            },
            {
                "block": {
                    "x_1": 185.12234497070312,
                    "y_1": 1012.4102783203125,
                    "x_2": 1459.57861328125,
                    "y_2": 1586.699951171875
                },
                "text": "Australia\nAustria\nBelgium\nCanada\nDenmark\nGermany\nItaly\nMexico\nPakistan\nPhilippines\nPoland\nRepublic of Ireland\nSingapore\nSwitzerland\nUnited Kingdom\n\nUnited States of America\n\ngpt-4-1106-preview\n\n \n\ngpt-4 gpt-3.5-turbo davinci-002\n16.83 47.87\nTAI 15.32 40.40\n935 16.13 42.65\n\na a a ce\n\n9.09\n\n11.77\n8.61\n6.29\n\ni in nt)\n\n \n\n \n\n \n\n8.16\n7.87\n10.32\n\n \n\n \n\n \n\nbebbaze 002\n\ner\n47.98\n47.03\n46.77\n\n \n\n \n\n \n\n11.03 14.71\n11.69 17.25 | 4492\n9.09 15.62 38.07\n9.33 13.14 mE\n5 16.54 42.67\n| oe 18.41\n12.70 15.56\n\n    \n  \n\n\u2014 ILBI\ng\nP3534 asa\n\n40.60\n\n \n\n \n\n \n\f",
                "id": null,
                "type": "table",
                "parent": null,
                "next": null,
                "score": 0.9775229096412659
            },
            {
                "block": {
                    "x_1": 185.82717895507812,
                    "y_1": 277.4165954589844,
                    "x_2": 1470.08544921875,
                    "y_2": 863.2593994140625
                },
                "text": "gpt-4-1106-preview gpt-4 gpt-3.5-turbo davinci-002 babbage-002\nCn\n\n \n\n \n\n    \n\n \n\n \n\n \n\nAustralia 11318 26.57 27.63\nAustria \u2014 7. = a [\nBelgium CA 26.96\nCanada \u2014 | _ 2938\nDenmark 133 \u2014, EE EE\nGermany I 26.97\nItaly \"6.16 \u2014 3 i Es\nMexico 1.73 24.10\nPakistan L 8.97 l 748 25.18 27.03\nPhilippines 5.62 71.74 Eu 21.20\nPoland 6.52 [ 26.00\nRepublic of Ireland | mG J 5.13 24.83 25.00\nSingapore 7.09 | re 26.40\nSwitzerland ; 9.16 2077] 26.83\nUnited Kingdom ie 26:94 26.54\nUnited States of America ff ] 7Al Bia 28.06 |\n\n \n\n \n\n   \n\n \n\n \n\f",
                "id": null,
                "type": "table",
                "parent": null,
                "next": null,
                "score": 0.9746643900871277
            }
        ],
        "page_data": {}
    },
    {
        "_blocks": [
            {
                "block": {
                    "x_1": 213.20016479492188,
                    "y_1": 928.7798461914062,
                    "x_2": 1438.25,
                    "y_2": 1995.08203125
                },
                "text": "gpt-4-1106-preview gpt-4 gpt-3.5-turbo davinci-002 babbage-002\nAtlanta Hawks 4.27 Zz ]\n6 4.09 : 13.07 17.23 14.40\n\n \n\nBoston Celtics\n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n  \n\n \n\n \n\n \n\nBrooklyn Nets 4.78 14.41\nCharlotte Hornets 4.35 12.53 14.33\nChicago Bulls\nCleveland Cavaliers 4.28 12.69 16.88 14.00\nDallas Mavericks 4.29 17.14\nDenver Nuggets 4.51 I\nDetroit Pistons 4.37\nGolden State Warriors 4.41 17.34 15.45\nHouston Rockets 3.99 13.39 14.30\nIndiana Pacers 3.96 12.65 14.24\nLos Angeles Clippers 4.7\nLos Angeles Lakers 4.38\nMemphis Grizzlies 4.18 12.20 14.35\nMiami Heat 4.30 14.40\nMilwaukee Bucks 4.27\nMinnesota Timberwolves 5,72 4.53 17.05 15.20\nNew Orleans Pelicans 5.89 | 5.00 | 14.37\nNew York Knicks 5.85 | 4.82 | : | 17.17\nOklahoma City Thunder 6.05 4.23 12.43 | o 14.39\nOrlando Magic 5.62 4.17 14.11 17.36\nPhiladelphia 76ers |\nPhoenix Suns i725 15.46\nPortland Trail Blazers 17.14\nSacramento Kings 17.35\nSan Antonio Spurs 172.\nToronto Raptors ES\nUtah Jazz 17.41\nWashington Wizards 17.15\n\n \n\f",
                "id": null,
                "type": "table",
                "parent": null,
                "next": null,
                "score": 0.9905263185501099
            },
            {
                "block": {
                    "x_1": 453.0599365234375,
                    "y_1": 737.1349487304688,
                    "x_2": 1195.132568359375,
                    "y_2": 768.3743896484375
                },
                "text": "Figure 7: Movie KG EASY-level zero sense rate grouped by years.\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.9796587228775024
            },
            {
                "block": {
                    "x_1": 483.35345458984375,
                    "y_1": 2038.5638427734375,
                    "x_2": 1163.6058349609375,
                    "y_2": 2070.19287109375
                },
                "text": "Figure 8: NBA EASY-level zero sense rate grouped by teams\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.9574530124664307
            },
            {
                "block": {
                    "x_1": 182.9237823486328,
                    "y_1": 287.5469970703125,
                    "x_2": 1444.9405517578125,
                    "y_2": 691.5851440429688
                },
                "text": "gpt-4-1106-preview gpt-4 gpt-3.5-turbo davinci-002 babbage-002\n\n  \n\n \n\n2015 9.80 17.86 20.00 26.00\n2016 16.33 7.69 29.41 xe 25.00\n2017 8.00 22.64 EE\n\n2018 14.29 7.55 26.67\n2019 15.91 5.88 a 19.44 21.88\n2020 15.56 11.63 26.32 20.93 20.00\n\n2021 12.99 7.89 27.78\n2022 16.22 27.50 22.03\n\n2023 Ew\n\f",
                "id": null,
                "type": "table",
                "parent": null,
                "next": null,
                "score": 0.8999383449554443
            }
        ],
        "page_data": {}
    },
    {
        "_blocks": [
            {
                "block": {
                    "x_1": 188.73023986816406,
                    "y_1": 614.6553955078125,
                    "x_2": 1453.237060546875,
                    "y_2": 1673.4000244140625
                },
                "text": "gpt-4-1106-preview gpt-4 gpt-3.5-turbo davinci-002 babbage-002\nAtlanta Hawks\n\n  \n  \n  \n  \n\n \n\nBoston Celtics 7.82 4.91 12.68\nBrooklyn Nets P1720\n\nCharlotte Hornets 7.40 4.98 ; 13.12\nChicago Bulls i\n\n \n\nCleveland Cavaliers 7.56\n\nDallas Mavericks\nDenver Nuggets\nDetroit Pistons\nGolden State Warriors [\n\n \n\n \n\n   \n\n  \n\nHouston Rockets 7.58 4.93\nIndiana Pacers 6.90 4.70 12.96\nLos Angeles Clippers\nLos Angeles Lakers\nMemphis Grizzlies 7.01 4.47\nMiami Heat BE\n\n    \n  \n\nMilwaukee Bucks\nMinnesota Timberwolves iq\nNew Orleans Pelicans\nNew York Knicks fi\nOklahoma City Thunder 6.89 4.44\nOrlando Magic a:\nPhiladelphia 76ers\nPhoenix Suns\nPortland Trail Blazers\nSacramento Kings\nSan Antonio Spurs\nToronto Raptors\nUtah Jazz\nWashington Wizards\n\n \n\n \n\n \n\n \n\n \n\n37.09\n\n37.29\n\n   \n\f",
                "id": null,
                "type": "table",
                "parent": null,
                "next": null,
                "score": 0.9927361011505127
            },
            {
                "block": {
                    "x_1": 480.6800231933594,
                    "y_1": 1706.331787109375,
                    "x_2": 1169.1995849609375,
                    "y_2": 1737.7305908203125
                },
                "text": "Figure 9: NBA HARD-level zero sense rate grouped by teams\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.9654659032821655
            }
        ],
        "page_data": {}
    },
    {
        "_blocks": [
            {
                "block": {
                    "x_1": 848.905517578125,
                    "y_1": 787.3970947265625,
                    "x_2": 1011.527587890625,
                    "y_2": 829.7904052734375
                },
                "text": "3 Method\n\f",
                "id": null,
                "type": "title",
                "parent": null,
                "next": null,
                "score": 0.9830922484397888
            },
            {
                "block": {
                    "x_1": 193.8446044921875,
                    "y_1": 193.1114501953125,
                    "x_2": 444.44891357421875,
                    "y_2": 235.40496826171875
                },
                "text": "2 Related Work\n\f",
                "id": null,
                "type": "title",
                "parent": null,
                "next": null,
                "score": 0.9708197712898254
            },
            {
                "block": {
                    "x_1": 188.8274383544922,
                    "y_1": 1545.1795654296875,
                    "x_2": 812.2052612304688,
                    "y_2": 2152.576416015625
                },
                "text": "On the other hand, knowledge graphs have the\nadvantages of customization to specific domains,\nevolving knowledge, and reduced potential for test\nset leakage, which has been employed as a struc-\ntured knowledge source for LLMs (Lin et al., 2019;\nAgarwal et al., 2020; Rosset et al., 2020) and also\nbeen employed as a tool to probe knowledge in\nLLMs. LAMA (Petroni et al., 2019) is the first\nwork to probe a pretrained model with KGs, where\nthey use the KG to generate the cloze statement and\nevaluate the LM\u2019s knowledge with accuracy. How-\never the cloze statement is not a natural question,\nand the correct answer is not unique in many cases,\nmaking the evaluation inaccurate. LPAQA (Jiang\net al., 2020) propose to paraphrase the cloze prompt\nand combine the answers, but still suffer from the\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.9375641942024231
            },
            {
                "block": {
                    "x_1": 840.1248779296875,
                    "y_1": 197.37460327148438,
                    "x_2": 1464.3829345703125,
                    "y_2": 766.3025512695312
                },
                "text": "same problem. In addition, these methods mainly\nfocus on accuracy but neglect that LLMs may re-\nspond differently to the same fact, where reliability\nshould also be considered. KaRR (Dong et al.,\n2023) proposes to solve this issue by using multi-\nple prompts for each KG edge and using the output\nlogits of LLMs to measure the knowledge reliabil-\nity. However, KaRR is inefficient for large graphs,\nand it is not generalizable due to the unavailable\nof LLM output logits. Moreover, transforming KG\ntriplets into questions is more natural than the text\ncloze task, but previous works mainly adopt the\ntext cloze task for simplicity. Finally, to our best\nknowledge, there is no existing work that visualizes\nthe LLM\u2019s knowledge with KG.\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.9337612986564636
            },
            {
                "block": {
                    "x_1": 848.6828002929688,
                    "y_1": 2073.524658203125,
                    "x_2": 1457.8250732421875,
                    "y_2": 2150.0302734375
                },
                "text": "The estimation of the posterior {a;, Bj }y, is\ndone in an iterative manner based on the outcome\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.9222240447998047
            },
            {
                "block": {
                    "x_1": 191.6542205810547,
                    "y_1": 264.26904296875,
                    "x_2": 809.7529907226562,
                    "y_2": 601.4021606445312
                },
                "text": "It\u2019s an established fact that pre-trained models have\nthe ability to learn and retain knowledge. For exam-\nple, Petroni et al. (2019) discovered that BERT (De-\nvlin et al., 2018), even without finetuning, har-\nbors relational knowledge comparable to traditional\nNLP methods. With LLMs showcasing superior\nin-context learning and knowledge retention, eval-\nuating their knowledge becomes pivotal to bolster\nperformance and mitigate hallucination.\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.9208331108093262
            },
            {
                "block": {
                    "x_1": 838.2959594726562,
                    "y_1": 843.2208251953125,
                    "x_2": 1464.7017822265625,
                    "y_2": 1338.372314453125
                },
                "text": "Our framework is shown in Figure 1. Given a\nknowledge graph, we first transform it into the pa-\nrameterized KG and initialize its parameters. Then\nwe sample a batch of edges for question gener-\nation based on the parameters of the KG. After\nthat, an LLM will be evaluated with the generated\nquestions, and the signal will be collected for each\nquestion via answer verification. For each signal\nwe collected from this Q&A process, we propa-\ngate and update the parameters based on the KG\nstructure. We iterate this process until the running\nmetrics are converged. Finally, we visualize and\nanalysis the updated PKG to gather the results.\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.8786518573760986
            },
            {
                "block": {
                    "x_1": 190.49403381347656,
                    "y_1": 582.4749755859375,
                    "x_2": 805.967041015625,
                    "y_2": 1557.0804443359375
                },
                "text": "performance and mitigate nallucination.\n\nThe knowledge assessment often tests the model\nwith specific knowledge-related datasets (Petroni\net al., 2020; Roberts et al., 2020; Peng et al., 2023;\nPress et al., 2022; Mallen et al., 2023). How-\never, given the fact that LLMs are trained on web-\ncrawled corpora and the data is constantly evolv-\ning, it is hard to exclude the test examples from\nthe pretraining corpus. For example, Deng et al.\n(2023) use fill-in probing and multi-choice prob-\ning to check the data leakage of pretrained LLMs.\nTheir results show that GPT-3.5-turbo exhibited\na noteworthy ability to guess the missing option.\nAnother concern is that the knowledge is dynamic,\nand the evaluation datasets remain fixed, which\nmakes it challenging to evaluate the LLMs\u2019 knowl-\nedge accurately. Dhingra et al. (2022) propose\na diagnostic dataset that pairs the text and times-\ntamp together and jointly models text and time.\nHowever, their dataset is static and designed for\n2010 to 2020, which is not suitable for evaluating\nthe LLMs\u2019 knowledge in the future. Finally, the\npredominant metric employed by these datasets\nrevolves around the test set accuracy, making it\nchallenging to identify solutions for enhancing the\nLLM and reducing the hallucination.\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.8410519957542419
            },
            {
                "block": {
                    "x_1": 849.0916748046875,
                    "y_1": 1569.3416748046875,
                    "x_2": 1463.4814453125,
                    "y_2": 1939.3551025390625
                },
                "text": "Intuitively, if an LLM failed in answering a ques-\ntion, there is a higher chance that the LLM also\nlacks knowledge of the related topics. To reflect\nthis inductive bias, we propose a parameterized\nKG, by augmenting each edge (s;,p;,0;) of the\noriginal KG with an additional error probability 6;\nreflecting the probability that an LLM may fail on\nthis edge. We use beta distribution to model @ due\nto the conjugacy between Bernoulli distribution\nand Beta distribution.\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.8257032036781311
            }
        ],
        "page_data": {}
    },
    {
        "_blocks": [
            {
                "block": {
                    "x_1": 307.79150390625,
                    "y_1": 197.33302307128906,
                    "x_2": 1336.6680908203125,
                    "y_2": 686.7883911132812
                },
                "text": " \n\ncount = 2,0 = 0.49, a = 2.67, B = 2.69\n\nQ: What is the central bank in Germany?\n\nA: The central bank in Germany is called the Deutsche\nBundesbank\n\nS: Correct\n\nQ: What is the name of the central bank in Germany?\nA: The central bank of Germany is the Deutsche\nBundesbank\n\nS: Correct\n\f",
                "id": null,
                "type": "figure",
                "parent": null,
                "next": null,
                "score": 0.9933413863182068
            },
            {
                "block": {
                    "x_1": 190.23165893554688,
                    "y_1": 716.6311645507812,
                    "x_2": 1464.424072265625,
                    "y_2": 780.8259887695312
                },
                "text": "Figure 2: Updated parameterized knowledge graph. The color of the edges represents the probability of the LLM to\nanswer the question correctly. We log the interactions between KGLENS and the LLM.\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.983502209186554
            },
            {
                "block": {
                    "x_1": 853.0213623046875,
                    "y_1": 837.7306518554688,
                    "x_2": 1381.8546142578125,
                    "y_2": 877.3322143554688
                },
                "text": "3.2 Graph-guided Question Generation\n\f",
                "id": null,
                "type": "title",
                "parent": null,
                "next": null,
                "score": 0.9770838618278503
            },
            {
                "block": {
                    "x_1": 217.39447021484375,
                    "y_1": 1942.044921875,
                    "x_2": 792.1573486328125,
                    "y_2": 2033.67333984375
                },
                "text": "aj = a; + [(response is incorrect) + Mj, (2\n\n \n\n8; = 8; + (response is correct) + Nj, 6)\n\f",
                "id": null,
                "type": "list",
                "parent": null,
                "next": null,
                "score": 0.9668512344360352
            },
            {
                "block": {
                    "x_1": 836.1917114257812,
                    "y_1": 1329.2010498046875,
                    "x_2": 1460.0169677734375,
                    "y_2": 1891.0423583984375
                },
                "text": "Each tuple can be transformed into a question by\nasking if the subject\u2019s relation is the object. But in\nthis way, the answer would be Yes for all the edges.\nTo formulate hard negative examples, we build a\nground truth answer set T; for each (s;,p;), and\nthe candidate answer set CG for each p;. Both T;\n\nand C; are derived from ihe full Wikidata insowl-\nedge eraph to ensure the completeness. Then, for a\ntuple {(s;,p;,0;) }, we use 0; to constitute the Yes\nquestion, and sample a random 0, from C; \u2014 T;\nto formulate the No question. Considering out\nQG process is on-the-fly during the evaluation,\nKGLENS can formulate different QA pairs for the\nsame fact. The sampling rate between yes and no\nquestion is evenly split, with a 50-50 distribution.\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.9430925250053406
            },
            {
                "block": {
                    "x_1": 194.577392578125,
                    "y_1": 2074.48193359375,
                    "x_2": 803.2907104492188,
                    "y_2": 2152.534912109375
                },
                "text": "where M; = _|incorrect neighborhood edges|\nand N; = |correct neighborhood edges].\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.9215187430381775
            },
            {
                "block": {
                    "x_1": 841.1532592773438,
                    "y_1": 883.8425903320312,
                    "x_2": 1464.5250244140625,
                    "y_2": 1267.271240234375
                },
                "text": "We use GPT-4 to transform the sampled edge K;\ninto the natural questions with few-shot in-context\nlearning. The prompts and demonstrations are\nshown in Appendix 6.2. We design two types of\nquestions for KGLENS: Yes/No Questions (judge-\nment) and Wh-Questions (generative), where the\nquestion type is controlled by the graph struc-\nture (out degree). In addition, to reduce the am-\nbiguity of entities, we provide the entity alias for\nquestion generation.\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.9123331904411316
            },
            {
                "block": {
                    "x_1": 849.107666015625,
                    "y_1": 1282.6683349609375,
                    "x_2": 1168.9788818359375,
                    "y_2": 1324.663818359375
                },
                "text": "3.2.1 Yes/No Questions\n\f",
                "id": null,
                "type": "title",
                "parent": null,
                "next": null,
                "score": 0.9065466523170471
            },
            {
                "block": {
                    "x_1": 846.9934692382812,
                    "y_1": 1912.64208984375,
                    "x_2": 1129.7772216796875,
                    "y_2": 1954.3326416015625
                },
                "text": "3.2.2 Wh-Questions\n\f",
                "id": null,
                "type": "title",
                "parent": null,
                "next": null,
                "score": 0.893147349357605
            },
            {
                "block": {
                    "x_1": 196.855224609375,
                    "y_1": 848.1400756835938,
                    "x_2": 811.665283203125,
                    "y_2": 876.480224609375
                },
                "text": "from the LLM. This process resembles the PageR-\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.8347835540771484
            },
            {
                "block": {
                    "x_1": 193.0657196044922,
                    "y_1": 1598.0289306640625,
                    "x_2": 806.7577514648438,
                    "y_2": 1897.63720703125
                },
                "text": "In order to account for the high correlation in\nerror probability among the connected edges, we\nhave additionally propagate the signal to the neigh-\nboring edges. Specifically, the signal gathered from\nPp; is propagated to both the incoming and outgoing\nedges that are connected to node s; and 0;. To\noptimize the computational process, we restrict the\nsignal propagation to one degree. Specifically,\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.8313969969749451
            },
            {
                "block": {
                    "x_1": 197.92970275878906,
                    "y_1": 1215.458740234375,
                    "x_2": 809.67626953125,
                    "y_2": 1370.753173828125
                },
                "text": "The top-n edges are then sent to LLM for exam-\nination and verification. The signal regarding the\ncorrectness of the output from LLM is collected for\neach of the edges accordingly.\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.8293870687484741
            }
        ],
        "page_data": {}
    },
    {
        "_blocks": [
            {
                "block": {
                    "x_1": 192.28651428222656,
                    "y_1": 1248.858154296875,
                    "x_2": 427.0558776855469,
                    "y_2": 1292.808837890625
                },
                "text": "4 Experiments\n\f",
                "id": null,
                "type": "title",
                "parent": null,
                "next": null,
                "score": 0.983111560344696
            },
            {
                "block": {
                    "x_1": 194.85806274414062,
                    "y_1": 2074.939697265625,
                    "x_2": 810.071533203125,
                    "y_2": 2150.2392578125
                },
                "text": "The statistics of our KGs are shown in Table 1.\nThe term \u201cdead edges\u201d refers to edges that are\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.9779376983642578
            },
            {
                "block": {
                    "x_1": 197.1905059814453,
                    "y_1": 413.0667724609375,
                    "x_2": 802.2386474609375,
                    "y_2": 452.85076904296875
                },
                "text": "3.3. QA Examination and Answer Verification\n\f",
                "id": null,
                "type": "title",
                "parent": null,
                "next": null,
                "score": 0.9596914052963257
            },
            {
                "block": {
                    "x_1": 838.6063842773438,
                    "y_1": 200.15626525878906,
                    "x_2": 1461.817626953125,
                    "y_2": 654.3697509765625
                },
                "text": "less intriguing to inquire about but are still crucial\nfor displaying entity relations. For example, cer-\ntain predicates such as \u201cmember of\u201d, \u201cdomestic\nrelation\u2019, or \u201ccontains the administrative territo-\nrial entity\u201d, exemplify links between entities, but\nthey are less captivating to inquire about and are\ntoo prevalent. Conversely, significant and mean-\ningful edges are referred to as \u201cactive edges\u201d, and\nwe use them to generate questions. Active edges\nrepresent the essential and noteworthy connections\nin the knowledge graph, from which we extract\ninformation to formulate insightful questions.\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.9491773247718811
            },
            {
                "block": {
                    "x_1": 187.52291870117188,
                    "y_1": 694.349853515625,
                    "x_2": 801.9404296875,
                    "y_2": 1219.1593017578125
                },
                "text": "To verify the response, we guide the LLMs to\ngenerate either \u201cYes\u201d or \u201cNo\u201d at the beginning\nof the response for Yes/No Questions and subse-\nquently generate accompanying explanations. This\napproach facilitates a straightforward verification\nprocess by examining the correspondence of the\ninitial word. For Wh-Questions, we instruct the\nLLM to list all the correct answers. In this case,\nthe assessment of the answer cannot be done by\nstring matching. Therefore, we employ a GPT-4\nmodel to determine the correctness of a response\ngiven the question, the ground truth objects and the\nalias. The prompts and demonstrations are listed in\nAppendix 6.2.\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.9411194324493408
            },
            {
                "block": {
                    "x_1": 869.3048095703125,
                    "y_1": 1566.4473876953125,
                    "x_2": 1461.12841796875,
                    "y_2": 1894.8997802734375
                },
                "text": "1. Win rate: LLM wins if the number of correct\nanswers surpasses the number of incorrect an-\nswers. The win rate signifies the portion of\nwinning edges out of all the examined edges.\n\n2. Zero sense rate: LLM has zero sense about\na certain knowledge if the model has never\nanswered the edge correctly.\n\f",
                "id": null,
                "type": "list",
                "parent": null,
                "next": null,
                "score": 0.9361687302589417
            },
            {
                "block": {
                    "x_1": 841.5051879882812,
                    "y_1": 658.2633056640625,
                    "x_2": 1467.1134033203125,
                    "y_2": 1216.7017822265625
                },
                "text": "Developing and cleaning these topic KG is not\ntrivial. The graph begins with a set of human-\ndefined central points (such as country names, NBA\nteam names), from which random walks are ini-\ntiated to growing the KG. After that, entity and\npredicate filters are applied. We keep entities men-\ntioned in multiple languages, as they often hold\nbroader significance. Frequently occurring entities\nare also given prominence due to their centrality\nwithin various contexts. Additionally, we exclude\nentities without aliases to focus on well-recognized\nand frequently referenced entities. The number of\npredicates in a KG is not significant, so we manu-\nally examine the KG predicates to exclude trivial,\nmalformed, or ambiguous ones.\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.9336180686950684
            },
            {
                "block": {
                    "x_1": 848.4688110351562,
                    "y_1": 1396.0230712890625,
                    "x_2": 1091.224365234375,
                    "y_2": 1436.5579833984375
                },
                "text": "4.2. Main Results\n\f",
                "id": null,
                "type": "title",
                "parent": null,
                "next": null,
                "score": 0.9068560600280762
            },
            {
                "block": {
                    "x_1": 838.7080078125,
                    "y_1": 1921.5062255859375,
                    "x_2": 1452.296142578125,
                    "y_2": 2151.158203125
                },
                "text": "Table 2 and Table 3 show results of the two met-\nrics over different knowledge graphs under EASY\nand HARD evaluation modes. Across varying dif.\nficulty levels, knowledge graphs, and the tested\nmodels, GPT-4 consistently outperforms the others\nin both metrics. Also, we find the recent released\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.9026128649711609
            },
            {
                "block": {
                    "x_1": 188.1458282470703,
                    "y_1": 198.17466735839844,
                    "x_2": 813.8766479492188,
                    "y_2": 386.0307312011719
                },
                "text": "Question and it makes no sense to check if a model\ncan enumerate all of them correctly. In KGLENS,\nwe opt to generate Wh-Questions only when the\nout degree of an entity is less than 10. Otherwise,\nthe Yes/No Questions prompt is adopted.\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.8573248982429504
            },
            {
                "block": {
                    "x_1": 191.2384033203125,
                    "y_1": 1803.844482421875,
                    "x_2": 810.0645751953125,
                    "y_2": 2074.897705078125
                },
                "text": "We prepare the testing knowledge graphs with\nWikidata Query Web Service in three topics: coun-\ntry, NBA, and movie. The country KG includes\nknowledge about 16 hand-picked countries. The\nNBA KG contains the knowledge related to 30\nNBA teams. And the movies are sampled from\nfilms that premiered after 2015.\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.8418628573417664
            },
            {
                "block": {
                    "x_1": 191.65377807617188,
                    "y_1": 469.8853454589844,
                    "x_2": 811.5595703125,
                    "y_2": 695.46337890625
                },
                "text": "We design the QA testing under two different diffi-\nculty levels: EASY and HARD. For EASY testing,\nwe only use Yes/No Questions to test the LLMs.\nFor HARD testing, we generate each type of ques-\ntion at a 50% chance. We use few-shot in-context\nlearning to test the LLMs.\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.8408126831054688
            },
            {
                "block": {
                    "x_1": 195.63259887695312,
                    "y_1": 1755.7078857421875,
                    "x_2": 638.050537109375,
                    "y_2": 1796.59375
                },
                "text": "4.1 Building Knowledge Graphs\n\f",
                "id": null,
                "type": "title",
                "parent": null,
                "next": null,
                "score": 0.8341997265815735
            },
            {
                "block": {
                    "x_1": 837.9710083007812,
                    "y_1": 1219.5294189453125,
                    "x_2": 1461.9986572265625,
                    "y_2": 1368.8177490234375
                },
                "text": "As a result the parameterized KGs concentrates\non specific domains and contains only relevant, sig-\nnificant knowledge as defined above. More details\nof KG construction are provided in the Appendix 6.\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.8087225556373596
            }
        ],
        "page_data": {}
    },
    {
        "_blocks": [
            {
                "block": {
                    "x_1": 544.9921264648438,
                    "y_1": 392.9801330566406,
                    "x_2": 1107.7083740234375,
                    "y_2": 426.0895690917969
                },
                "text": "Table 1: Statistics of the testing knowledge graphs.\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.9975382089614868
            },
            {
                "block": {
                    "x_1": 341.9006652832031,
                    "y_1": 802.1778564453125,
                    "x_2": 1310.6436767578125,
                    "y_2": 832.8605346679688
                },
                "text": "Table 2: Win rate results for different LLMs evaluated under EASY and HARD modes.\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.9936368465423584
            },
            {
                "block": {
                    "x_1": 334.618408203125,
                    "y_1": 869.97265625,
                    "x_2": 1326.2252197265625,
                    "y_2": 1182.510498046875
                },
                "text": " \n\n \n\n \n\nLLMs Country NBA Movie\nEASY HARD EASY HARD EASY HARD\nBabbage-002 24.51 51.56 15.34 38.61 26.70 56.77\nDavinci-002 24.44 47.27 17.69 37.89 28.54 52.71\nGPT-3.5-turbo 14.98 20.32 17.17 21.09 22.70 29.36\nGPT-4 742 12.99 6.07 8.13 8.35 17.67\nGPT-4-1106-preview 7.59 14.16 8.19 12.42 9.21 21.43\n\n \n\f",
                "id": null,
                "type": "table",
                "parent": null,
                "next": null,
                "score": 0.99137282371521
            },
            {
                "block": {
                    "x_1": 327.0179443359375,
                    "y_1": 461.38836669921875,
                    "x_2": 1325.2579345703125,
                    "y_2": 774.5775146484375
                },
                "text": " \n\n \n\n \n\nLLMs Country NBA Movie\nEASY HARD EASY HARD EASY HARD\nBabbage-002 57.46 34.39 58.32 27.65 57.48 31.00\nDavinci-002 58.85 38.36 58.21 30.57 55.66 34.72\nGPT-3.5-turbo 74.43 63.42 57.98 56.95 62.80 57.70\nGPT-4 84.79 74.06 84.23 78.93 85.14 70.80\nGPT-4-1106-preview 82.27 72.42 79.09 70.57 83.15 66.95\n\n \n\f",
                "id": null,
                "type": "table",
                "parent": null,
                "next": null,
                "score": 0.9882707595825195
            },
            {
                "block": {
                    "x_1": 301.9342041015625,
                    "y_1": 1208.8353271484375,
                    "x_2": 1342.234375,
                    "y_2": 1240.7698974609375
                },
                "text": "Table 3: Zero sense rate results for different LLMs evaluated under EASY and HARD modes.\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.977764904499054
            },
            {
                "block": {
                    "x_1": 192.88682556152344,
                    "y_1": 1275.7139892578125,
                    "x_2": 804.94970703125,
                    "y_2": 2137.296875
                },
                "text": "GPT-4-1106-preview performs worse than GPT-\n4, which is reasonable for a preview version. We\nfind the gap betwen GPT-3.5-turbo and GPT-4 is\nlarge across all domains and all difficulty levels,\nand GPT-3.5-turbo is even worse than the legacy\nLLMs under NBA KG EASY mode. Upon inves-\ntigating the evaluation logs, GPT-3.5 exhibits a\nconservative approach, abstaining from generating\nanswers when lacking confidence rather than pro-\nviding speculative responses. Responses following\nthis protocol consistently begin with the phrases,\n\u201cT am sorry, but I couldn\u2019t find any information\non/about...\u201d, \u201cI\u2019m sorry, but as an AI assistant, I\ndo not have the capability to provide real-time in-\nformation ...\u201d. In such cases, the edge would be\nmarked as failed when the model declines to an-\nswer a question. In contrast, we did not observe\nsuch behavior in any of the other models. It seems\nas though GPT-3.5-turbo could have performed bet-\nter if there were no restrictions in place. Lastly,\nwe find the two legacy models exhibit compara-\nble performance across evaluations. The random\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.9597455263137817
            },
            {
                "block": {
                    "x_1": 418.41015625,
                    "y_1": 199.78317260742188,
                    "x_2": 1219.987548828125,
                    "y_2": 367.5631408691406
                },
                "text": " \n\nKG Active Edges Dead Edges Nodes Predicates\n\n \n\nCountry 7844 9441 12760 338\nNBA 2689 1158 805 57\nMovies 8704 3053 7965 340\n\f",
                "id": null,
                "type": "table",
                "parent": null,
                "next": null,
                "score": 0.9338853359222412
            },
            {
                "block": {
                    "x_1": 835.1986083984375,
                    "y_1": 1528.192138671875,
                    "x_2": 1455.5419921875,
                    "y_2": 1750.5340576171875
                },
                "text": "In addition to the quantitative analysis, we have\nidentified four categories of common errors within\nLLMs: Factual errors, Obsolete Knowledge er-\nrors, Self-contradiction errors, and Inconsistent Re-\n\nsponse errors. We provide examples of each error\ntype in Table 4.\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.931414008140564
            },
            {
                "block": {
                    "x_1": 835.658935546875,
                    "y_1": 1301.2633056640625,
                    "x_2": 1455.3948974609375,
                    "y_2": 1526.0699462890625
                },
                "text": "guessing baseline of the win rate is 50% for EASY\nevaluation, and 25% for HARD evaluation. We\nfind Babbage-002 and Davinci-002 results are just\nslightly better than the random guessing, clearly\nshowing the gap between the legacy LLMs and the\ncurrent LLMs.\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.9280850291252136
            },
            {
                "block": {
                    "x_1": 846.5040893554688,
                    "y_1": 1773.90185546875,
                    "x_2": 1176.0245361328125,
                    "y_2": 1811.9713134765625
                },
                "text": "4.3 Focus of Evaluation\n\f",
                "id": null,
                "type": "title",
                "parent": null,
                "next": null,
                "score": 0.8854632377624512
            },
            {
                "block": {
                    "x_1": 851.2509155273438,
                    "y_1": 1824.0057373046875,
                    "x_2": 1462.723876953125,
                    "y_2": 1917.99365234375
                },
                "text": "In this section, we show KGLENS can be used for\ndifferent focuses of evaluation, such as temporal\n\neee set metab creme\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.8476473689079285
            },
            {
                "block": {
                    "x_1": 831.5341796875,
                    "y_1": 1999.8636474609375,
                    "x_2": 1460.966796875,
                    "y_2": 2147.784423828125
                },
                "text": "We first show the HARD mode Movie KG results\nwhich are grouped by the movie release years in\nFigure 3. The EASY mode results are in Ap-\npendix 7. It should be noted that it is reasonable\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.845733642578125
            },
            {
                "block": {
                    "x_1": 850.0892333984375,
                    "y_1": 1686.2391357421875,
                    "x_2": 1465.4368896484375,
                    "y_2": 1713.3756103515625
                },
                "text": "sponse errors. We provide examples of each error\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.8375670909881592
            },
            {
                "block": {
                    "x_1": 848.7223510742188,
                    "y_1": 1831.5126953125,
                    "x_2": 1461.3018798828125,
                    "y_2": 1861.291259765625
                },
                "text": "In this section, we show KGLENS can be used for\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.8281863927841187
            }
        ],
        "page_data": {}
    },
    {
        "_blocks": [
            {
                "block": {
                    "x_1": 846.5769653320312,
                    "y_1": 1357.855712890625,
                    "x_2": 1128.209228515625,
                    "y_2": 1395.9930419921875
                },
                "text": "4.3.2 Entity Groups\n\f",
                "id": null,
                "type": "title",
                "parent": null,
                "next": null,
                "score": 0.9904378056526184
            },
            {
                "block": {
                    "x_1": 246.75299072265625,
                    "y_1": 197.6053924560547,
                    "x_2": 1432.3480224609375,
                    "y_2": 664.8499755859375
                },
                "text": " \n\nError Type\n\nQuestion\n\nResponse\n\nAnswer\n\n \n\nFactual Error\n\nOn which side does the railway\ntraffic run in Israel?\n\nRailway traffic in Israel runs on\nthe right-hand side.\n\nLeft side\n\n \n\nWho is the current head of state\n\nThe current head of state in Eng-\n\nCharles III of the\n\n \n\n \n\nObsolete Knowledge in England? land is Queen Elizabeth II. United Kingdom\n\nIs the Australian dollar the cur- je te erolen date\nSelf-contradiction rency of Nauru (also known as | aINGSlaBd watt y he Pz > Yes\n\nNR)? a sma island nation in the Pa-\n\n\u00b0 cific Ocean.\n\nIn Tonga (also known as TO), do No: inTbaws. people diveron\nInconsistent Response people drive on the right side of h 7 ft sid & f i Nad No\n\nthe road? the left side of the road.\n\nIs the left the driving side in No, the right is the driving side Yes\n\nTonga (also known as TO)?\n\nin Tonga.\n\n \n\f",
                "id": null,
                "type": "table",
                "parent": null,
                "next": null,
                "score": 0.9892382621765137
            },
            {
                "block": {
                    "x_1": 531.3023681640625,
                    "y_1": 688.646240234375,
                    "x_2": 1117.34619140625,
                    "y_2": 723.1251831054688
                },
                "text": "Table 4: Error types uncovered from the country KG.\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.9666160345077515
            },
            {
                "block": {
                    "x_1": 197.8870391845703,
                    "y_1": 1232.8629150390625,
                    "x_2": 1457.2239990234375,
                    "y_2": 1299.0225830078125
                },
                "text": "Figure 3: Zero sense rate grouped by years for moive KG in HARD mode. The three recent LLMs perform worse\nfor knowledge after 2020, while the behaviors of the two legacy LLMs are more randomly.\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.9635803699493408
            },
            {
                "block": {
                    "x_1": 848.4854125976562,
                    "y_1": 1551.0313720703125,
                    "x_2": 1464.1968994140625,
                    "y_2": 1954.3145751953125
                },
                "text": "The proficiency levels across countries can be vi-\nsualized using a color coded table, where a darker\ncolor signifies higher zero sense rate and thus a\nlower level of proficiency. Taking GPT-4 evaluated\nagainst country KG under HARD level difficulty\nfor example, GPT-4 exhibits a recognition accuracy\nwhere the Austria, Mexico, and Italy are identified\nand ranked as 1, 2, and 3 respectively. In con-\ntrast, countries such as Canada, Philippines, and\nthe United Kingdom are positioned at the lower\nend of the ranking scale.\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.9368468523025513
            },
            {
                "block": {
                    "x_1": 192.0645294189453,
                    "y_1": 1348.2032470703125,
                    "x_2": 810.162841796875,
                    "y_2": 2076.205078125
                },
                "text": "that the rankings in Figure 3 are not strictly aligned\nwith the years, as the temporal difference is not\nthe only factor that affect the evaluation results.\nFrom this figure, we observe that both the GPT-3.5\nand GPT-4 perform worse for questions after 2020,\nwhich is reasonable as they were mainly pretrained\nwith data before September 2021. Also, we found\nthat GPT-4 significantly outperform the other mod-\nels in terms of zero-sense rate and win rate. All\nmodels get worse when evaluated in HARD mode,\nbut GPT-3.5 is more robust. This is because a big\nportion of GPT-3.5\u2019s failure cases are caused by re-\nfusing to answer the questions, instead of providing\nwrong answers, which explains its results in EASY\nand HARD testing. Interestingly, we find all three\nrecent LLMs perform worse for movies released in\n2018, which might related to the pretraining data\ncollection but need further investigation as their\npretraining data are not publicly available.\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.9148442149162292
            },
            {
                "block": {
                    "x_1": 173.8511505126953,
                    "y_1": 773.4020385742188,
                    "x_2": 1462.6187744140625,
                    "y_2": 1195.2880859375
                },
                "text": "2015\n2016\n2017\n2018\n2019\n2020\n2021\n2022\n2023\n\n \n\n \n\n \n\n \n\n \n\ngpt-4-1106-preview gpt-4\n21.05 LSrO2 22722)\n2993) 2222 31.11\n19.44 18.87\nEx | ey\n25.58\n26.67\n\n \n\n \n\n   \n\ngpt-3.5-turbo davinci-002 babbage-002\n\nI 58.70\n54.84\n\n58.70\n51.35\n55.81\n51.52\n54.05\n\na\n\f",
                "id": null,
                "type": "table",
                "parent": null,
                "next": null,
                "score": 0.9103956818580627
            },
            {
                "block": {
                    "x_1": 849.8111572265625,
                    "y_1": 1959.9718017578125,
                    "x_2": 1466.234130859375,
                    "y_2": 2149.99853515625
                },
                "text": "The rationale behind the ranking can be eluci-\ndated by examining the dotted heatmap 4. In figure,\nthe size of each dot corresponds to the number of\nedges within the predicate sub-group, normalized\nby the total size of edges in the entire group. Addi-\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.8392323851585388
            },
            {
                "block": {
                    "x_1": 862.2994995117188,
                    "y_1": 1888.2679443359375,
                    "x_2": 1463.75244140625,
                    "y_2": 1915.703857421875
                },
                "text": "he United Kingdom are positioned at the lower\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.82813560962677
            }
        ],
        "page_data": {}
    },
    {
        "_blocks": [
            {
                "block": {
                    "x_1": 188.87498474121094,
                    "y_1": 196.44520568847656,
                    "x_2": 1462.875,
                    "y_2": 1558.989990234375
                },
                "text": "9}e1 BSUDS O19Z\no +\n\n1.0\n0.8\n0.2\n0.0\n\nSo S\n\n7\u00b0 @ e @ eeeeeeceese e@@e \u201c@-e*ee@ ,\nY\n\u00b0 ee . \u201c@-re-e o \u00ab @ \u00ae@ -@e@ee -&\n&\noc err @ \u00b0\u00b0 @ e ee eo @+ ese @ce@ .%,\n7-27 @ \u00b0 e \u00b0 ee o ee 441 Oe 9 4-@- O%\n- @@ oe 6 oe \u201c+ @ + \u00a9 \u00a9 \u00a9 \u00a9 so Oe \u00a9e@eree .\u201d\n%\ne @ \u00b0 o 8 6 . @ ee \u201c27+ @@eer @ee@ee Go,\n~@, %\nee +s e . - e 7 OO es sere e-eeeee -%,%\n0, 7\ne \u00b0 @-- eee e oO Os 85%\n0)\n\u00b0 \u00b0 oe @ \u00b0 e e 7+ \u00b0@@-@ .%\n<\u00a2\n\u00b0 ee e e oe @e ee \u00a9 oe G09 ee ee Oe @ Mis 5)\nQe %%,%,\n\u2014s 2 @ @ Oe \u00ab \u00a9 @ \u00a9 \u00a9 er->ce@@e0e ~@% %\nXO\noe e e \u00b0 e . ee -e@eece .%\n0,\noe ee oe e e eecee @- seeece (%,\nY\nOn, %\n1b gt@ eit 4 \u00b0 oe ec oe oc oor re See Qe\nYY\nYe\ne\n\n \n\nayeo1paig\n\nCountry\n\f",
                "id": null,
                "type": "figure",
                "parent": null,
                "next": null,
                "score": 0.9949016571044922
            },
            {
                "block": {
                    "x_1": 189.34523010253906,
                    "y_1": 1586.22021484375,
                    "x_2": 1460.6195068359375,
                    "y_2": 1682.1385498046875
                },
                "text": "Figure 4: Predicate level knowledge proficiency of GPT-4 evaluated under HARD difficulty. The darker color\nindicates a lower zero sense rate. The dot size shows the proportional size of the number of edges in the predicate\nsub-group.\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.9751127362251282
            },
            {
                "block": {
                    "x_1": 851.3441772460938,
                    "y_1": 1968.6470947265625,
                    "x_2": 1061.9735107421875,
                    "y_2": 2009.7027587890625
                },
                "text": "5 Conclusion\n\f",
                "id": null,
                "type": "title",
                "parent": null,
                "next": null,
                "score": 0.9654274582862854
            },
            {
                "block": {
                    "x_1": 846.1884155273438,
                    "y_1": 2036.603271484375,
                    "x_2": 1461.3299560546875,
                    "y_2": 2145.87939453125
                },
                "text": "In this work, we introduced KGLENS, a novel and\nefficient method tailored for visualizing and eval-\nuating the factual knowledge embedded in LLMs.\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.8517206907272339
            },
            {
                "block": {
                    "x_1": 189.4140625,
                    "y_1": 2000.0216064453125,
                    "x_2": 809.4835815429688,
                    "y_2": 2149.1025390625
                },
                "text": "Concentrating on the Austria and the Canada,\nwhich represent the highest and lowest ranked coun-\ntries, respectively, within the context of our inves-\ntigation, it becomes evident that GPT-4 exhibits\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.8252824544906616
            },
            {
                "block": {
                    "x_1": 840.4601440429688,
                    "y_1": 1749.3111572265625,
                    "x_2": 1465.042236328125,
                    "y_2": 1934.41943359375
                },
                "text": "enhanced proficiency pertaining to specific predi-\ncate sub-groups. Notably, these sub-groups include\n\u201clocated in time zone\u201d, \u201clocated in the administra-\ntive territorial entity\u201d, \u201celectrical plug type,\u201d \u201c\u201cemer-\ngency phone number,\u201d and \u201chead of state\u201d.\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.81918865442276
            }
        ],
        "page_data": {}
    },
    {
        "_blocks": [
            {
                "block": {
                    "x_1": 196.49273681640625,
                    "y_1": 823.9208984375,
                    "x_2": 350.3165588378906,
                    "y_2": 862.4317016601562
                },
                "text": "References\n\f",
                "id": null,
                "type": "title",
                "parent": null,
                "next": null,
                "score": 0.9851135015487671
            },
            {
                "block": {
                    "x_1": 855.0430297851562,
                    "y_1": 201.47523498535156,
                    "x_2": 1461.9312744140625,
                    "y_2": 322.7075500488281
                },
                "text": "Zhengbao Jiang, Frank F Xu, Jun Araki, and Graham\nNeubig. 2020. How can we know what language\nmodels know? Transactions of the Association for\nComputational Linguistics, 8:423-438.\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.9512779712677002
            },
            {
                "block": {
                    "x_1": 189.1377410888672,
                    "y_1": 198.6692352294922,
                    "x_2": 805.443115234375,
                    "y_2": 755.8172607421875
                },
                "text": "By evaluating various LLMs with our developed\ndomain-specific KGs, we show KGLENSprovides\nadaptable and customizable views of an LLM\u2019s\nknowledge. In addition to evaluating the accuracy\nof facts, our proposed parameterized KG offers\nan efficient way to assess the knowledge reliabil-\nity of LLMs. Our evaluation is more similar to\nhuman interaction compared to the widely used\ntext cloze task. Furthermore, our assessment KGs,\nsourced from Wikidata, encompass over 700 rela-\ntions and 21K entities. They are made available\nto the research community, fostering collaboration\nand serving as a valuable resource for future in-\nvestigations into language models and knowledge\nrepresentation.\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.938774049282074
            },
            {
                "block": {
                    "x_1": 196.66062927246094,
                    "y_1": 1229.339111328125,
                    "x_2": 809.8153686523438,
                    "y_2": 1349.1832275390625
                },
                "text": "Jacob Devlin, Ming-Wei Chang, Kenton Lee, and\nKristina Toutanova. 2018. Bert: Pre-training of deep\nbidirectional transformers for language understand-\ning. arXiv preprint arXiv: 1810.04805.\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.9260966181755066
            },
            {
                "block": {
                    "x_1": 199.20626831054688,
                    "y_1": 879.9078979492188,
                    "x_2": 809.0790405273438,
                    "y_2": 1227.9498291015625
                },
                "text": "Oshin Agarwal, Heming Ge, Siamak Shakeri, and\nRami Al-Rfou. 2020. Knowledge graph based syn-\nthetic corpus generation for knowledge-enhanced\nlanguage model pre-training. arXiv preprint\narXiv:2010.12688.\n\nChunyuan Deng, Yilun Zhao, Xiangru Tang, Mark Ger-\nstein, and Arman Cohan. 2023. Benchmark probing:\nInvestigating data leakage in large language models.\nIn NeurIPS 2023 Workshop on Backdoors in Deep\nLearning - The Good, the Bad, and the Ugly.\n\f",
                "id": null,
                "type": "list",
                "parent": null,
                "next": null,
                "score": 0.9194331169128418
            },
            {
                "block": {
                    "x_1": 193.21096801757812,
                    "y_1": 1990.2135009765625,
                    "x_2": 810.700927734375,
                    "y_2": 2113.607177734375
                },
                "text": "Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou,\nMantas Mazeika, Dawn Song, and Jacob Steinhardt.\n2020. Measuring massive multitask language under-\nstanding. arXiv preprint arXiv:2009.03300.\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.8765647411346436
            }
        ],
        "page_data": {}
    },
    {
        "_blocks": [
            {
                "block": {
                    "x_1": 198.30880737304688,
                    "y_1": 511.7725830078125,
                    "x_2": 991.0870361328125,
                    "y_2": 553.2116088867188
                },
                "text": "6.1.1 Sampling Strategies and Preserving Data Distribution\n\f",
                "id": null,
                "type": "title",
                "parent": null,
                "next": null,
                "score": 0.9959523677825928
            },
            {
                "block": {
                    "x_1": 194.92312622070312,
                    "y_1": 1403.3240966796875,
                    "x_2": 696.0040283203125,
                    "y_2": 1444.3201904296875
                },
                "text": "6.1.3 Filtering Less Relevant Entities\n\f",
                "id": null,
                "type": "title",
                "parent": null,
                "next": null,
                "score": 0.9951364398002625
            },
            {
                "block": {
                    "x_1": 196.9431610107422,
                    "y_1": 278.8352355957031,
                    "x_2": 630.7440185546875,
                    "y_2": 317.66534423828125
                },
                "text": "6.1 Knowledge Graph Cleaning\n\f",
                "id": null,
                "type": "title",
                "parent": null,
                "next": null,
                "score": 0.9944747090339661
            },
            {
                "block": {
                    "x_1": 192.94479370117188,
                    "y_1": 976.3975219726562,
                    "x_2": 676.6199951171875,
                    "y_2": 1016.4320068359375
                },
                "text": "6.1.2 Focus and Curated Relevance\n\f",
                "id": null,
                "type": "title",
                "parent": null,
                "next": null,
                "score": 0.9768879413604736
            },
            {
                "block": {
                    "x_1": 189.0432586669922,
                    "y_1": 1467.399169921875,
                    "x_2": 1457.3944091796875,
                    "y_2": 1622.864013671875
                },
                "text": "The other challenges we encounter in Wikidata pertains to the noise within its knowledge graph. This\nnoise manifests in the form of entities that are rarely mentioned or of lesser importance in the context of\n\nour research objectives. To maintain the integrity of our analysis, it is important to identify and filter out\nthese less relevant entities.\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.9323519468307495
            },
            {
                "block": {
                    "x_1": 191.2984619140625,
                    "y_1": 578.849365234375,
                    "x_2": 1458.156494140625,
                    "y_2": 804.4707641601562
                },
                "text": "Maintaining the original data distribution is important when cleaning a knowledge graph. To achieve this,\nrandom walk with both forward and backward dimension are employed. Sorting by random value of each\nqueried edges, the sub-knowledge graph contains the representative samples that mirror the diversity of\nthe original knowledge graph, we can preserve the inherent distribution of entities and relationships. This\napproach guarantees that our cleaned knowledge graph remains a faithful representation of the underlying\ndata, enabling us to draw accurate conclusions from our research.\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.8787064552307129
            },
            {
                "block": {
                    "x_1": 190.78575134277344,
                    "y_1": 811.7435302734375,
                    "x_2": 1458.8031005859375,
                    "y_2": 917.4686279296875
                },
                "text": "The extent of the random walk distance is flexible and tailored to specific requirements. Within our sub\nknowledge graphs, we conduct random walks spanning three steps, encompassing both nodes and edges\nwithin this range for analysis.\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.8292150497436523
            },
            {
                "block": {
                    "x_1": 184.59291076660156,
                    "y_1": 1046.7901611328125,
                    "x_2": 1463.294921875,
                    "y_2": 1203.4259033203125
                },
                "text": "In the realm of knowledge graphs, Wikidata stands out as a repository of extensive information. However,\nour research necessitates a more nuanced approach. While Wikidata offers comprehensive knowledge, our\nfocus lies in curated topics and entities tailored for specific purposes. This distinction is vital as it allows\nus to delve deeper into specialized domains, ensuring the precision and relevance of the data we analyze.\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.8224852085113525
            },
            {
                "block": {
                    "x_1": 196.4093780517578,
                    "y_1": 197.60537719726562,
                    "x_2": 385.989990234375,
                    "y_2": 235.72369384765625
                },
                "text": "6 Appendix\n\f",
                "id": null,
                "type": "title",
                "parent": null,
                "next": null,
                "score": 0.8162198662757874
            },
            {
                "block": {
                    "x_1": 193.2881622314453,
                    "y_1": 343.8875732421875,
                    "x_2": 1457.30615234375,
                    "y_2": 461.5101318359375
                },
                "text": "Given Wikidata\u2019s vastness and inherent noise, we implement multiple strategies to maintain focus,\nrelevance, and precision. Following techniques empower us to delve into specialized domains and ensure\nus a targeted and reliable exploration of the data.\n\f",
                "id": null,
                "type": "text",
                "parent": null,
                "next": null,
                "score": 0.8121863007545471
            },
            {
                "block": {
                    "x_1": 233.1582794189453,
                    "y_1": 2034.4742431640625,
                    "x_2": 1459.1927490234375,
                    "y_2": 2149.1796875
                },
                "text": "\u00a2 Filtering out entities with no alias: entities without aliases are less likely to be widely recognized\nor referenced. By excluding these entities, we focus our analysis on well-known and frequently\nmentioned entities, aligning our research with more meaningful and impactful data points.\n\f",
                "id": null,
                "type": "list",
                "parent": null,
                "next": null,
                "score": 0.8027685880661011
            }
        ],
        "page_data": {}
    }
]
