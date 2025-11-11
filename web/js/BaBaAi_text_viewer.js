import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

app.registerExtension({
    name: "Comfy.BaBaAiTextViewer",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "BaBaAiTextViewer" || nodeData.name === "BaBaAiConcatTextViewer") {
            const outputWidgetId = "text_output_widget";

            function getOutputWidget(node) {
                return node.widgets?.find(w => w.name === outputWidgetId);
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
                    outputWidget.serialize = false;
                }

                if (!text || !text.length) {
                    outputWidget.value = "";
                } else {
                    const formattedText = Array.isArray(text) ? text.join('\n') : String(text);
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
            nodeType.prototype.onConfigure = function (info) {
                if (onConfigure) {
                    onConfigure.apply(this, arguments);
                }
                
                requestAnimationFrame(() => {

                    populate.call(this, [""]);
                    
                    this.setDirtyCanvas(true, true);
                });
            };

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);
                
                populate.call(this, [""]);
                
                requestAnimationFrame(() => {
                    const outputWidget = getOutputWidget(this);
                    if (outputWidget && this.widgets) {
                        const currentIndex = this.widgets.indexOf(outputWidget);
                        if (currentIndex !== -1 && currentIndex !== this.widgets.length - 1) {
                            this.widgets.splice(currentIndex, 1);
                            this.widgets.push(outputWidget);
                        }
                    }
                });
            };
            
            const onSerialize = nodeType.prototype.onSerialize;
            nodeType.prototype.onSerialize = function(o) {
                const r = onSerialize ? onSerialize.apply(this, arguments) : o;
                
                if (this.widgets && o.widgets_values) {
                    const outputWidget = getOutputWidget(this);
                    if (outputWidget) {
                        const outputIndex = this.widgets.indexOf(outputWidget);
                        if (outputIndex !== -1) {
                            o.widgets_values = this.widgets
                                .filter((w, i) => i !== outputIndex && w.serialize !== false)
                                .map(w => w.value);
                        }
                    }
                }
                
                return r;
            };
            
            const onDeserialize = nodeType.prototype.onDeserialize;
            nodeType.prototype.onDeserialize = function(info) {
                if (onDeserialize) {
                    onDeserialize.apply(this, arguments);
                }
                
                if (info.widgets_values && this.widgets) {
                    const outputWidget = getOutputWidget(this);
                    if (outputWidget) {
                        const normalWidgets = this.widgets.filter(w => w !== outputWidget && w.serialize !== false);
                        normalWidgets.forEach((w, i) => {
                            if (i < info.widgets_values.length) {
                                w.value = info.widgets_values[i];
                            }
                        });
                    }
                }
            };
        }
    },
});