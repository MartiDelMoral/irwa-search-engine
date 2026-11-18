# Information Retrieval and Web Analytics (IRWA) - Final Project template

<table>
  <tr>
    <td style="vertical-align: top;">
      <img src="static/image.png" alt="Project Logo"/>
    </td>
    <td style="vertical-align: top;">
      This repository contains the template code for the IRWA Final Project - Search Engine with Web Analytics.
      The project is implemented using Python and the Flask web framework. It includes a simple web application that allows users to search through a collection of documents and view analytics about their searches.
    </td>
  </tr>
</table>

----
## Project Structure

```
/irwa-search-engine
├── myapp                # Contains the main application logic
├── templates            # Contains HTML templates for the Flask application
├── static               # Contains static assets (images, CSS, JavaScript)
├── data                 # Contains the dataset file (fashion_products_dataset.json)
├── project_progress     # Contains your solutions for Parts 1, 2, and 3 of the project
├── .env                 # Environment variables for configuration (e.g., API keys)
├── .gitignore           # Specifies files and directories to be ignored by Git
├── LICENSE              # License information for the project
├── requirements.txt     # Lists Python package dependencies
├── web_app.py           # Main Flask application
└── README.md            # Project documentation and usage instructions
```


----
## To download this repo locally

Open a terminal console and execute:
```
cd <your preferred projects root directory>
git clone https://github.com/trokhymovych/irwa-search-engine.git
```

## Setting up the Python environment (only for the first time you run the project)
### Install virtualenv
Setting up a virtualenv is recommended to isolate the project dependencies from other Python projects on your machine.
It allows you to manage packages on a per-project basis, avoiding potential conflicts between different projects.

In the project root directory execute:
```bash
pip3 install virtualenv
virtualenv --version
```

### Prepare virtualenv for the project
In the root of the project folder run to create a virtualenv named `irwa_venv`:
```bash
virtualenv irwa_venv
```

If you list the contents of the project root directory, you will see that it has created a new folder named `irwa_venv` that contains the virtualenv:
```bash
ls -l
```

The next step is to activate your new virtualenv for the project:
```bash
source irwa_venv/bin/activate
```

or for Windows...
```cmd
irwa_venv\Scripts\activate.bat
```

This will load the python virtualenv for the project.

### Installing Flask and other packages in your virtualenv
Make sure you are in the root of the project folder and that your virtualenv is activated (you should see `(irwa_venv)` in your terminal prompt).
And then install all the packages listed in `requirements.txt` with:
```bash
pip install -r requirements.txt
```

If you need to add more packages in the future, you can install them with pip and then update `requirements.txt` with:
```bash
pip freeze > requirements.txt
```

Enjoy!


## Starting the Web App
From the repository root, activate your virtual environment as described above,
then start Flask:
```powershell
python web_app.py
```
The app starts at:
```
http://localhost:8088/
```

Choose a ranking method on the home page and enter a query. All four options
first require every cleaned query term to occur in a product (AND semantics).
Word2Vec trains on the product corpus the first time that option is used.
The search page works without a Groq API key; only its optional generated
summary needs that service.

## Part 3: Ranking experiments

The search form lets you compare the following rankers on the same query:

| Method | How it ranks | Strengths | Limitations |
| --- | --- | --- | --- |
| TF-IDF cosine | Cosine similarity between query and product TF-IDF vectors. | Familiar, fast, and rewards informative terms. | Term-frequency saturation is not modeled; long product text can dilute its score. |
| BM25 | Okapi BM25 with `k1=1.5` and `b=0.75`, including document-length normalization. | Usually handles repeated terms and long documents better than basic TF-IDF. | `k1` and `b` are tunable; it uses exact terms and is not semantic. |
| Custom product score | `0.70 × normalized BM25 + 0.15 × rating/5 + 0.10 × discount/100 + 0.05 × in-stock`. | Combines text relevance with product quality and purchase availability. | The weights are hand chosen; popularity/rating can bias results, and missing values contribute zero. |
| Word2Vec cosine | Cosine similarity between averaged query and product word vectors. | Can capture some contextual similarity between words. | Vectors are trained on this corpus; rare or weakly related words may have poor vectors. AND matching still requires every query term literally. |

The custom score clamps rating to 0–5 and discount to 0–100 before
normalization. A missing numeric value contributes zero. The ranking method can
be changed in the home-page selector without changing the query.

TF-IDF and BM25 return the same candidate set because both use AND matching,
but can order those products differently. TF-IDF cosine uses raw term counts
weighted by IDF and normalizes by vector length. BM25 instead has diminishing
returns for repeated terms and adjusts document length relative to the corpus
average. Neither method understands synonyms.

To print the top 20 products for each of the five Part 2 queries using all four
methods, run this from the repository root:

```powershell
python -m myapp.search.run_part3
```

Word2Vec vectors use the mean of each product's processed word vectors, just
like the query vector uses the mean of its processed word vectors. Training is
deterministic and happens once per search-engine instance.

To compare the evaluation metrics on the supplied validation queries and
judgments, run:

```powershell
python -m myapp.search.evaluation
```

This prints Precision, Recall, Average Precision, F1, NDCG, MAP, and MRR for
each method. Use the same queries and labels across methods when comparing
their scores.


## Creating your own GitHub repo
After creating the project and code in local computer...

1. Login to GitHub and create a new repo.
2. Go to the root page of your new repo and note the url from the browser.
3. Execute the following locally:
```bash
cd <project root folder>
git init -b main
git add . && git commit -m "initial commit"
git remote add origin <your GitHub repo URL from the browser>
git push -u origin main
```

## Usage: 
0. Put the data file `fashion_products_dataset.json` in the `data` folder. It will be provided to you by the instructor.
1. As for Parts 1, 2, and 3 of the project, please use the `project_progress` folder to store your solutions. Each part should contain `.pdf` file with your report and `.ipynb` (Jupyter Notebook) file with your code for solution and `README.md` with explanation of the content and instructions for results reproduction.
2. For the Part 4, of the project, you should build a web application using Flask that allows users to search through a collection of documents and view analytics about their searches. You should work mailnly in the `web_app.py` file `myapp` and `templates` folders. Feel free to change any code or add new files as needed. The provided code is just a starting point to help you get started quickly.
3. Make sure to update the `.env` file with your Groq API key (can be found [here](https://groq.com/), the free version is more than enough for our purposes) and any other necessary configurations. IMPORTANT: Do not share your `.env` file publicly as it contains sensitive information. It is included in `.gitignore` to prevent accidental commits. (It should never be included in the repos and appear here only for demonstration purposes).
4. Have fun and be creative!

## Attribution:
The project is adapted from the following sources:
- [IRWA Template 2021](https://github.com/irwa-labs/search-engine-web-app)
