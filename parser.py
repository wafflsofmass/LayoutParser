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
        
