This project is meant for human learning robot research done as a senior project. The goal is to use
LLM to generate JSON schemas that will then map to Webots API in order to simulate actual robotic actions.

Unless you have a really good GPU, this cannot be run locally. The code was built using the school's HPC.
Login instructions are on the school website. 

Certain models requires different libraries that need to be locally downloaded. Because of this, it is
recommended to use the conda environment.

# Setting Up HPC

Refer to the school's website to login. Once logged in:

The node that you are on is called the login node. I'm not sure if this is the official term, but this
is the node that has internet but *no* GPU. This means you can't run any model/code available in this repo
on this node.
On top of that, the GPU nodes *don't* have internet but to run any model from Hugging Face, you need to be
able to access the API (which requires internet). The solution is to download the repo by using
`git clone` **in the login node**. If you try doing `git clone` on a GPU node, it will not work
because it requires internet.

`module avail` command lists all of the available modules that the HPC offers. You can think of
these modules as regular apps on your computer, like Notebook or Word. You cannot download anything outside
of these modules unless you contact whoever is in charge. For this research though, we only need
Anaconda, so download whatever is the latest version of Anaconda using `module load [exact_module_name]`.

**Note:** conda was already working once logged in but in case it doesn't, make sure to load it.

Create a regular conda environment. This is googleable. 
**Note:** Once inside the conda environment, always download using `conda forge`. However, there
are some libraries that isn't available to conda. In this case, it's OK to download with pip. However,
you cannot download anything using sudo apt.

The HPC comes with a folder called scratch. This is the folder where you will put ***all*** your work. 
If you want to know why, google it. Make sure that at this point, you are in your conda environment
and in the scratch directory.

Now, you're ready to code!

# Running Hugging Face Models

To run any models given the script in the corresponding directory, you ***must*** locally download the
model in the same directory as the script. The command to do is `huggingface-cli download <model-id> --local-dir <target-folder>`.

After that, you run the script by doing `./[script_name].py --model-dir-name [dir_name]`, where `dir_name` is 
the name in which you downloaded the model.


