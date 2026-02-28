# Getting started with the labs

## Setting Up the Config File

### Steps to Follow:

1. **Download the following compressed folders course portal:**
   - `trees_dataset.tar.gz`
   - `unsplash_50.tar.gz`


2. **Unzip the folders:**
   Extract the contents of the compressed files.


> if there are files starting with . like `._ww-4.jpg`, then remove such files. A simple command to get rid of all such files is: `find . -name ".*" ! -name "." ! -name ".." -delete`


3. **Update the configuration file:**
   Open the `config.yaml` file and set the paths to the datasets. Update it as follows:

```yaml
   datasets:
     unsplash: "path/to/unsplash_100"
     trees: "path/to/trees_dataset"
```

# Sequence of scripts

## Clip

1. run `src/svlearn_vlu/clip/embed_images.py`
2. run the notebook `docs/notebooks/clip.ipynb`

## Blip

1. run `src/svlearn_vlu/blip/captioner.py`
2. run the notebook `docs/notebooks/blip.ipynb`

## Blip2

1. run `src/svlearn_vlu/blip2/captioner.py`
2. run the notebook `docs/notebooks/blip-2.ipynb`

## LLaVa

1. run `src/svlearn_vlu/llava/captioner.py`
2. run the notebook `docs/notebooks/llava.ipynb`


## SigLip2

1. run `src/svlearn_vlu/siglip2/embed_images.py`
2. run the notebook `docs/notebooks/siglip2.ipynb`

## InsightFace

1. Create a directory data/images and put some images(with faces)
2. Create a directory data/faces and put reference face images
3. run the notebook `docs/notebooks/view_faces.ipynb`

## Deepseek-vl (Ignore this for now)

### **Setup**

Follow these steps to add deepseek-vl library to your environment. 

```bash
    # open a new terminal with no python environment active
    # create a new directory called DeepSeek-VL 
    # 1 Clone the repository
    git clone https://github.com/deepseek-ai/DeepSeek-VL

    # 2 Change directory to the cloned repo
    cd DeepSeek-VL

    # 3 Create a virtual environment with uv
    uv sync --upgrade

    #if step 3 fails because of the sentencepiece dependency, make the changes below

        # update the version of sentencepiece in the pyproject.toml like so 
        dependencies = [
            ...,
            "sentencepiece>=0.1.96",
            ...
        ]


        [project.optional-dependencies]
        gradio = [
            ...
            "SentencePiece>=0.1.96"
        ]



    # 4 Now return to the vision_language_understanding directory and activate the project enviornment
    cd /path/to/vision_language/understanding
    source .venv/bin/activate

    # 5 Add the cloned directory as a dependency
    uv pip install -e "deepseek-vl @ path/to/DeepSeek-VL"


```

1. run `src/svlearn_vlu/deepseek_vl/captioner.py`
2. run the notebook `docs/notebooks/deepseek_vl.ipynb`