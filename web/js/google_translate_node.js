import { app } from "../../scripts/app.js";

const findWidget = (node, name, attr = "name", func = "find") =>
  node.widgets[func]((w) => w[attr] === name);

app.registerExtension({
  name: "Comfy.GoogleTranslateNode",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    // --- GoogleTranslateNode
    if (
      nodeData.name == "GoogleTranslateTextNode" ||
      nodeData.name == "GoogleTranslateCLIPTextEncodeNode"
    ) {
      // Node Created - No button is added here anymore.
      const onNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        onNodeCreated?.apply?.(this, arguments);
      };

      // Node Configure
      const onConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function () {
        onConfigure?.apply(this, arguments);

        if (this?.widgets_values.length) {
          if (typeof this.widgets_values[2] === "string") {
            const customtext = findWidget(this, "text", "name", "findIndex");
            if (customtext !== -1 && this.widgets[customtext]) {
                this.widgets[customtext].value = this.widgets_values[2];
            }
            if (this.widgets[2]) {
                this.widgets[2].value = false;
            }
          }
        }
      };
    }

    // --- GoogleTranslateNode
  },
});