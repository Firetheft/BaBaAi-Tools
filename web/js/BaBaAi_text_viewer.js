import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

app.registerExtension({
    name: "Comfy.BaBaAiTextViewer",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "BaBaAiTextViewer" || nodeData.name === "BaBaAiConcatTextViewer") {
            const outputWidgetName = "text_output";
            const outputWidgetId = "text_output_widget";

            function getOutputWidget(node) {
                return node.widgets.find(w => w.name === outputWidgetId);
            }

            function populate(text) {
                const node = this;
                let outputWidget = getOutputWidget(node);

                if (!outputWidget) {
                    outputWidget = ComfyWidgets["STRING"](node, outputWidgetId, ["STRING", { multiline: true }], app).widget;
                    outputWidget.inputEl.readOnly = true;
                    outputWidget.inputEl.style.opacity = 0.6;
                    outputWidget.name = outputWidgetId;
                    outputWidget.label = "";
                }

                if (!text || !text.length) {
                    outputWidget.value = "";
                } else {
                    const formattedText = text.join('\n');
                    outputWidget.value = formattedText;
                }

                requestAnimationFrame(() => {
                    node.onResize?.(node.computeSize());
                    app.graph.setDirtyCanvas(true, false);
                });
            }

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                if (message.text_output) {
                    populate.call(this, message.text_output);
                }
            };
            
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                const configureArgs = arguments;
                requestAnimationFrame(() => {
                    onConfigure?.apply(this, configureArgs);
                    if (this.widgets_values?.length) {
                         const outputValue = this.widgets_values[this.widgets_values.length - 1];
                         if (outputValue) {
                            populate.call(this, outputValue);
                         }
                    }
                });
                return undefined;
            };

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);
                populate.call(this, [""]);
            };
        }
    },
});