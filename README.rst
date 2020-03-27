=========
pandalyse
=========


A toolkit for an analysis environment with pandas.



What is pandalyse
-----------------

Pandalyse offers an analysis environment and tools for pandas.
The main features are:

* **Selector**: Define and store cuts in a Selector object: `cutted_df = selector(df)`
* **Trainer**: Train multiple mva with scikit-learn interface in the way: `trainer.fit(signal_df, background_df)`
* **Analysis**: Store and retrive Selectors, Trainings, numpy-arrays and dataframes in predefined locations: `df = ana.data.get("MySignalData")`


Note
====

This project has been set up using PyScaffold 3.2.3. For details and usage
information on PyScaffold see https://pyscaffold.org/.
