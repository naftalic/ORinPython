#!/usr/bin/env python
# coding: utf-8

# # Cells
# A notebook is a list of cells. Cells contain either explanatory text or executable code and its output. Click a cell to select it.

# ## Code cells
# Below is a **code cell**. Once the toolbar button indicates CONNECTED, click in the cell to select it and execute the contents in the following ways:
# 
# * Click the **Play icon** in the left gutter of the cell;
# * Type **Cmd/Ctrl+Enter** to run the cell in place;
# * Type **Shift+Enter** to run the cell and move focus to the next cell (adding one if none exists); or
# * Type **Alt+Enter** to run the cell and insert a new code cell immediately below it.
# 
# There are additional options for running some or all cells in the **Runtime** menu.
# 

# In[1]:


a = 10
a


# ## Text cells
# This is a **text cell**. You can **double-click** to edit this cell. Text cells
# use markdown syntax. To learn more, see our [markdown
# guide](/notebooks/markdown_guide.ipynb).
# 
# You can also add math to text cells using [LaTeX](http://www.latex-project.org/)
# to be rendered by [MathJax](https://www.mathjax.org). Just place the statement
# within a pair of **\$** signs. For example `$\sqrt{3x-1}+(1+x)^2$` becomes
# $\sqrt{3x-1}+(1+x)^2.$
# 

# ## Adding and moving cells
# You can add new cells by using the **+ CODE** and **+ TEXT** buttons that show when you hover between cells. These buttons are also in the toolbar above the notebook where they can be used to add a cell below the currently selected cell.
# 
# You can move a cell by selecting it and clicking **Cell Up** or **Cell Down** in the top toolbar. 
# 
# Consecutive cells can be selected by "lasso selection" by dragging from outside one cell and through the group.  Non-adjacent cells can be selected concurrently by clicking one and then holding down Ctrl while clicking another.  Similarly, using Shift instead of Ctrl will select all intermediate cells.

# # Working with python
# Colaboratory is built on top of [Jupyter Notebook](https://jupyter.org/). Below are some examples of convenience functions provided.

# Long running python processes can be interrupted. Run the following cell and select **Runtime -> Interrupt execution** (*hotkey: Cmd/Ctrl-M I*) to stop execution.

# In[2]:


import time
print("Sleeping")
time.sleep(30) # sleep for a while; interrupt me!
print("Done Sleeping")


# ## System aliases
# 
# Jupyter includes shortcuts for common operations, such as ls:

# In[ ]:


get_ipython().system('ls /bin')


# That `!ls` probably generated a large output. You can select the cell and clear the output by either: 
# 
# 1. Clicking on the clear output button (x) in the toolbar above the cell; or
# 2. Right clicking the left gutter of the output area and selecting "Clear output" from the context menu.
# 
# Execute any other process using `!` with string interpolation from python variables, and note the result can be assigned to a variable:

# In[ ]:


message = 'Colaboratory is great!'
foo = get_ipython().getoutput("echo -e '$message\\n$message'")
foo


# ## Magics
# Colaboratory shares the notion of magics from Jupyter. There are shorthand annotations that change how a cell's text is executed. To learn more, see [Jupyter's magics page](http://nbviewer.jupyter.org/github/ipython/ipython/blob/1.x/examples/notebooks/Cell%20Magics.ipynb). 
# 

# In[ ]:


get_ipython().run_cell_magic('html', '', "<marquee style='width: 30%; color: blue;'><b>Whee!</b></marquee>")


# In[ ]:


get_ipython().run_cell_magic('html', '', '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 450 400" width="200" height="200">\n  <rect x="80" y="60" width="250" height="250" rx="20" style="fill:red; stroke:black; fill-opacity:0.7" />\n  <rect x="180" y="110" width="250" height="250" rx="40" style="fill:blue; stroke:black; fill-opacity:0.5;" />\n</svg>')


# ## Automatic completions and exploring code
# 
# Colab provides automatic completions to explore attributes of Python objects, as well as to quickly view documentation strings. As an example, first run the following cell to import the  [`numpy`](http://www.numpy.org) module.

# In[ ]:


import numpy as np


# If you now insert your cursor after `np` and press **Period**(`.`), you will see the list of available completions within the `np` module. Completions can be opened again by using **Ctrl+Space**.

# In[ ]:


np


# If you type an open parenthesis after any function or class in the module, you will see a pop-up of its documentation string:

# In[ ]:


np.ndarray


# The documentation can be opened again using **Ctrl+Shift+Space** or you can view the documentation for method by mouse hovering over the method name.
# 
# When hovering over the method name the `Open in tab` link will open the documentation in a persistent pane. The `View source` link will navigate to the source code for the method.

# ## Exception Formatting

# Exceptions are formatted nicely in Colab outputs:

# In[ ]:


x = 1
y = 4
z = y/(1-x)


# ## Rich, interactive outputs
# Until now all of the generated outputs have been text, but they can be more interesting, like the chart below. 

# In[ ]:


import numpy as np
from matplotlib import pyplot as plt

ys = 200 + np.random.randn(100)
x = [x for x in range(len(ys))]

plt.plot(x, ys, '-')
plt.fill_between(x, ys, 195, where=(ys > 195), facecolor='g', alpha=0.6)

plt.title("Fills and Alpha Example")
plt.show()


# # Integration with Drive
# 
# Colaboratory is integrated with Google Drive. It allows you to share, comment, and collaborate on the same document with multiple people:
# 
# * The **SHARE** button (top-right of the toolbar) allows you to share the notebook and control permissions set on it.
# 
# * **File->Make a Copy** creates a copy of the notebook in Drive.
# 
# * **File->Save** saves the File to Drive. **File->Save and checkpoint** pins the version so it doesn't get deleted from the revision history. 
# 
# * **File->Revision history** shows the notebook's revision history. 

# ## Commenting on a cell
# You can comment on a Colaboratory notebook like you would on a Google Document. Comments are attached to cells, and are displayed next to the cell they refer to. If you have **comment-only** permissions, you will see a comment button on the top right of the cell when you hover over it.
# 
# If you have edit or comment permissions you can comment on a cell in one of three ways: 
# 
# 1. Select a cell and click the comment button in the toolbar above the top-right corner of the cell.
# 1. Right click a text cell and select **Add a comment** from the context menu.
# 3. Use the shortcut **Ctrl+Shift+M** to add a comment to the currently selected cell. 
# 
# You can resolve and reply to comments, and you can target comments to specific collaborators by typing *+[email address]* (e.g., `+user@domain.com`). Addressed collaborators will be emailed. 
# 
# The Comment button in the top-right corner of the page shows all comments attached to the notebook.
