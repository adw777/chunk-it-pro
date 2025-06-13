Installation
============

Requirements
------------

* Python 3.8+
* API keys for embedding providers (OpenAI, VoyageAI, or both)

Install Dependencies
--------------------

.. code-block:: bash

   pip install -r requirements.txt

Environment Variables
---------------------

Create a `.env` file:

.. code-block:: bash

   OPENAI_API_KEY=your_openai_api_key_here
   VOYAGEAI_API_KEY=your_voyage_api_key_here
   OMNIPARSE_API_URL=your_omniparse_api_url_here

Verification
------------

Test your installation:

.. code-block:: bash

   python test_installation.py