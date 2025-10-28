import { app } from "../../../scripts/app.js";
import { $el } from "../../../scripts/ui.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

function createBabaColorWidget(node, inputName, inputData) {
    const defaultColor = inputData[1]?.default || "#ffffff";

    const getTextColorForBg = (hex) => {
        try {
            let r = 0, g = 0, b = 0;
            if (hex.length == 4) {
                r = parseInt(hex[1] + hex[1], 16);
                g = parseInt(hex[2] + hex[2], 16);
                b = parseInt(hex[3] + hex[3], 16);
            } else if (hex.length == 7) {
                r = parseInt(hex.substring(1, 3), 16);
                g = parseInt(hex.substring(3, 5), 16);
                b = parseInt(hex.substring(5, 7), 16);
            } else {
                return "#000000";
            }
            
            const luminance = (r * 299 + g * 587 + b * 114) / 1000;

            return luminance > 140 ? "#000000" : "#FFFFFF";
        } catch (e) {
            return "#000000";
        }
    };

    const widget = {
        type: "BABA_COLOR",
        name: inputName,
        _value: defaultColor,
        options: { default: defaultColor, ...(inputData[1] || {}) },
        y: 0,

        _value: defaultColor,
        options: { default: defaultColor, ...(inputData[1] || {}) },
        y: 0,

        _cachedBgColor: null,
        _cachedTextColor: null,

        get value() {
            return this._value;
        },
        set value(v) {
            const newValue = v || this.options.default;
            if (this._value !== newValue) {
                this._value = newValue;

                if (this.colorElement) {
                    this.colorElement.style.backgroundColor = newValue;
                }
                 if (this.textElement) {
                    this.textElement.textContent = newValue;
                }
            }
        },

        draw: function (ctx, node, widgetWidth, widgetY, height) {
            const margin = 10;
            const border = 1;
            const widgetHeight = ComfyWidgets.NODE_WIDGET_HEIGHT || 24;
            const drawY = widgetY + (height - widgetHeight) / 2;
            const drawX = margin;
            const drawWidth = widgetWidth - margin * 2;
            const drawHeight = widgetHeight;

            ctx.fillStyle = this.value || this.options.default;
            ctx.fillRect(drawX, drawY, drawWidth, drawHeight);
            
            ctx.strokeStyle = "#000";
            ctx.strokeRect(drawX, drawY, drawWidth, drawHeight);

            this.colorBoxArea = { x: drawX, y: drawY, w: drawWidth, h: drawHeight };

            ctx.font = "14px Arial";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            
            const currentBgColor = this.value || this.options.default;
            
            if (this._cachedBgColor !== currentBgColor) {
                this._cachedTextColor = getTextColorForBg(currentBgColor);
                this._cachedBgColor = currentBgColor;
            }
            
            ctx.fillStyle = this._cachedTextColor;

            const textX = widgetWidth / 2;
            const textY = drawY + drawHeight / 2;
            
            ctx.fillText(this.value || this.options.default, textX, textY);
            
            this.last_y = widgetY;
            this.last_h = height;
            this.last_w = widgetWidth;
        },

        mouse: function (event, pos, node) {
             if (pos[1] < this.last_y || pos[1] > this.last_y + this.last_h) {
                 return false;
             }

             const clickX = pos[0];
             const clickY = pos[1];

             if ( this.colorBoxArea &&
                  clickX >= this.colorBoxArea.x &&
                  clickX <= this.colorBoxArea.x + this.colorBoxArea.w &&
                  clickY >= this.colorBoxArea.y &&
                  clickY <= this.colorBoxArea.y + this.colorBoxArea.h)
             {
                if (event.type === LiteGraph.pointerevents_method + "down") {
                    this.openColorPicker(event, node);
                    event.stopPropagation();

                    return true;
                }
            }
            return false;
        },

        openColorPicker: function(event, node) {
            const picker = document.createElement("input");
            picker.type = "color";
            picker.value = this.value;

            Object.assign(picker.style, {
                position: "fixed",
                left: `${event.clientX}px`,
                top: `${event.clientY}px`,
                width: "0px", height: "0px", padding: "0px", border: "none", opacity: 0,
                pointerEvents: "none"
            });

            const onChange = (e) => {
                this.value = e.target.value;
                if (this.callback) this.callback(this.value);
                node.setDirtyCanvas(true, true);
            };

            const cleanup = () => {
                picker.removeEventListener("input", onChange);
                picker.removeEventListener("change", onChange);
                picker.removeEventListener("blur", cleanup);
                picker.remove();
            };

            picker.addEventListener("input", onChange);
            picker.addEventListener("change", onChange);
            picker.addEventListener("blur", cleanup);

            document.body.appendChild(picker);

            setTimeout(() => {
                try {
                     if (typeof picker.showPicker === 'function') {
                         picker.showPicker();
                     } else {
                         picker.click();
                     }

                     picker.focus();
                } catch (e) {
                     console.error("Error trying to open color picker:", e);
                     cleanup();
                }
            }, 50);
        },

        computeSize: function (width) {
            return [width, ComfyWidgets.NODE_WIDGET_HEIGHT || 24];
        },
    };

    return widget;
}

app.registerExtension({
    name: "Comfy.BaBaAi.ColorPickerWidget.CustomObject",
    getCustomWidgets: function(appInstance) {
        return {

            BABA_COLOR: (node, inputName, inputData) => {

                 const widget = createBabaColorWidget(node, inputName, inputData, appInstance);
                 node.addCustomWidget(widget);

                 return { widget: widget };
            }
        };
    },

    async beforeRegisterNodeDef(nodeType, nodeData, appInstance) {
            
        if (nodeData.name === "ColorPalettePickerNode") {

            const setupColorCaching = (node) => {
                const randomizeWidget = node.widgets.find(w => w.name === "randomize_colors");
                const colorWidgets = [
                    node.widgets.find(w => w.name === "color1"),
                    node.widgets.find(w => w.name === "color2"),
                    node.widgets.find(w => w.name === "color3"),
                    node.widgets.find(w => w.name === "color4"),
                    node.widgets.find(w => w.name === "color5"),
                ].filter(w => w);

                if (!node._originalBabaColors) {
                    node._originalBabaColors = colorWidgets.map(w => w.value);
                }
                
                colorWidgets.forEach((widget, index) => {
                    widget.callback = (newValue) => {
                        if (randomizeWidget && !randomizeWidget.value) {
                            if (node._originalBabaColors) {
                                node._originalBabaColors[index] = newValue;
                            }
                        }
                    };
                });

                if (randomizeWidget) {
                    randomizeWidget.callback = () => {
                        if (!randomizeWidget.value && node._originalBabaColors) {
                            colorWidgets.forEach((w, i) => w.value = node._originalBabaColors[i]);
                        }
                    };
                }
            };
            
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                
                if (message?.new_colors) {
                    const colorWidgets = [
                        this.widgets.find(w => w.name === "color1"),
                        this.widgets.find(w => w.name === "color2"),
                        this.widgets.find(w => w.name === "color3"),
                        this.widgets.find(w => w.name === "color4"),
                        this.widgets.find(w => w.name === "color5"),
                    ].filter(w => w);
                    
                    message.new_colors.forEach((color, i) => {
                        if (colorWidgets[i]) {
                            colorWidgets[i].value = color;
                        }
                    });
                    
                    if (this._originalBabaColors) {
                        this._originalBabaColors = message.new_colors.slice(0, 5);
                    }
                }
            };

            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function() {
                onConfigure?.apply(this, arguments);
                setTimeout(() => {
                    setupColorCaching(this);
                    appInstance.graph.setDirtyCanvas(true, true);
                }, 50);
            };

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                onNodeCreated?.apply(this, arguments);
                setTimeout(() => {
                    setupColorCaching(this);
                }, 50);
            };
        }
    }
});