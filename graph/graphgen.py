import pydot

__all__ = ["Graph"]

# code is from https://gist.github.com/apaszke/01aae7a0494c55af6242f06fad1f8b70
# reference: draw_net.py from caffe
# we combine these code to draw the structure of pytorch's model

class Graph(object):
    """
    draw structure of model
    """

    def __init__(self, name="resnet", rankdir="TB"):
        self.seen = set()
        self.dot = pydot.Dot(name, graph_type="digraph", rankdir=rankdir)
        self.style_params = {"shape": "octagon",
                             "fillcolor": "gray",
                             "style": "filled",
                             "label": "",
                             "color": "none"}
        self.style_layers = {"shape": "box",
                             "fillcolor": "blue",
                             "style": "filled",
                             "label": "",
                             "color": "none"}

        self.ignored_node = [
            "AccumulateGrad"
        ]
    def _add_nodes(self, var):
        """
        add node to the graph
        """
        if var not in self.seen:
            value = str(type(var).__name__)

            # remove "backward" from name string
            value = value.replace('Backward', '')

            # add label to nodes
            self.style_layers["label"] = value

            # assign color to different layers
            if "Conv" in value:
                self.style_layers["fillcolor"] = "cyan"
            elif "BatchNorm" in value:
                self.style_layers["fillcolor"] = "darkseagreen"
            elif "Threshold" in value:
                self.style_layers["fillcolor"] = "crimson"
                self.style_layers["label"] = "ReLU"
            elif "Add" in value:
                self.style_layers["fillcolor"] = "darkorchid"
            elif "Pool" in value:
                self.style_layers["fillcolor"] = "gold"
            elif "Linear" in value:
                self.style_layers["fillcolor"] = "chartreuse"
            elif "View" in value:
                self.style_layers["fillcolor"] = "brown"
            else:
                self.style_layers["fillcolor"] = "aquamarine"

            self.dot.add_node(pydot.Node(
                str(id(var)), **self.style_layers))

            self.seen.add(var)

            # get next node
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    # remove leaf nodes and specific layer
                    if u[0] is not None and str(type(u[0]).__name__) not in self.ignored_node :
                        self.dot.add_edge(pydot.Edge(
                            str(id(u[0])), str(id(var))))
                        self._add_nodes(u[0])

    def draw(self, var):
        """
        get structure of model
        :params var: torch.autograd.Variable
        variable in pytorch has its attribute <next_funciton> which is point to their creator function,
        if a variable is leaf node, its creator is None
        in order to create graph of the structure of a model, we first run forward of the model to get the output variable
        this output is the final node of whole network, starting from this node, we search every node on the network, so as to 
        get the graph of the model
        """
        self._add_nodes(var.grad_fn)

    def save(self, file_name="network.jpeg"):
        """
        save to file
        :params file_name: path to save
        """
        ext = file_name[file_name.rfind(".") + 1:]
        with open(file_name, 'wb') as fid:
            img = self.dot.create(format=ext)
            fid.write(img)
        print("|===>Save Network Graph Done! save_path:", file_name)
