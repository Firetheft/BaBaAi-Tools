/**
 * MultiAreaConditioning.js
 * Fully Compatible with ComfyUI v0.3.43 Frontend API
 *
 * @description 多区域条件控制的前端可视化界面，支持点击开关式拖拽交互 (Final Layout-Corrected Version)
 * @author Davemane42 (Corrected by Gemini)
 * @version 3.2.0
 * @updated 2025-08-09
 * @comfyui_version 0.3.43
 * @frontend_api_version 1.23.4
 */

import { app } from "/scripts/app.js";

// 常量定义
const CONSTANTS = {
    CANVAS_HEIGHT: 250, // 控制节点中间可视化操作区域（我们称之为“画布”）的高度
    WIDGET_HEIGHT: 30, // 控件滑块的高度。
    TITLE_HEIGHT: 70, // 这并不是节点标题的高度，而是定义了从节点最顶部到画布开始位置的垂直距离。为顶部的输入插槽（条件0, 条件1等）预留出70像素高的空间。
    GAP_BELOW_CANVAS: 10, // 定义了画布区域底部和下方第一个控件滑块顶部之间的垂直间距。
    MARGIN: 0, // 控制底部边距
    SIDE_MARGIN: 70, // 控制画布与节点左右两侧的距离
    GRID_SIZE: 64, // 定义了画布上网格线的间距。
    MIN_AREA_SIZE: 32, // 定义了在画布上可拖拽的色块区域的最小尺寸。
    DEFAULT_RESOLUTION: { width: 1024, height: 1024 }, // 默认分辨率
    COLORS: {
        BACKGROUND: "#404040", // 画布区域的深灰色背景色。
        BORDER: "#000000", // 画布区域外围的黑色边框色。
        GRID: "#606060", // 画布上网格线的颜色。
        SELECTED: "#8B5CF6", // 当前被选中的区域（高亮色块）以及左侧高亮圆圈的紫色填充色。
        SELECTED_BORDER: "#FFFFFF", // 当前被选中的区域外围的白色边框色。
        AREAS: ["#8B7355", "#7B8B55", "#5B8B75", "#6B5B8B"] // 一个颜色数组，用于显示其他未被选中的区域。代码会按顺序循环使用这些颜色来区分不同的区域。
    }
};

// 实用工具函数
const Utils = {
    createCustomInt: function(node, inputName, val, func, config = {}) {
        const defaultConfig = { min: 0, max: 4096, step: 640, precision: 0 };
        return node.addWidget("number", inputName, val, func, Object.assign({}, defaultConfig, config));
    },
    transformFunc: function(widget, value, node, index) {
        try {
            const s = widget.options.step / 10;
            widget.value = Math.round(value / s) * s;
            if (node.properties && node.properties["values"] && node.widgets && node.index !== undefined && node.widgets[node.index] && node.widgets[node.index].value !== undefined) {
                const selectedIndex = Math.round(node.widgets[node.index].value);
                if (selectedIndex >= 0 && selectedIndex < node.properties["values"].length) {
                    node.properties["values"][selectedIndex][index] = widget.value;
                }
            }
        } catch (error) {
            console.error("Parameter transformation error:", error);
        }
    }
};

// 布局管理器 (最终修正版)
const LayoutManager = {
    computeCanvasSize: function(node) {
        if (!node.widgets) return;

        const canvasWidget = node.widgets.find(w => w.type === "customCanvas");
        const otherWidgets = node.widgets.filter(w => w.type !== "customCanvas");

        // 1. 定位画布
        if (canvasWidget) {
            canvasWidget.y = CONSTANTS.TITLE_HEIGHT;
            canvasWidget.h = CONSTANTS.CANVAS_HEIGHT;
        }
        node.canvasHeight = CONSTANTS.CANVAS_HEIGHT;
        
        // 2. 从画布下方开始，从上往下定位所有其他控件
        let currentY = CONSTANTS.CANVAS_HEIGHT + CONSTANTS.GAP_BELOW_CANVAS;
        
        otherWidgets.forEach(widget => {
            widget.y = currentY;
            currentY += CONSTANTS.WIDGET_HEIGHT;
        });

        // 3. 根据内容自动计算并设置节点高度
        // 核心修正: 总高度不再包含TITLE_HEIGHT，因为框架会自动添加它。
        const totalHeight = CONSTANTS.CANVAS_HEIGHT + CONSTANTS.GAP_BELOW_CANVAS + (otherWidgets.length * CONSTANTS.WIDGET_HEIGHT) + CONSTANTS.MARGIN;
        if (node.size[1] !== totalHeight) {
            node.size[1] = totalHeight;
        }
    }
};

// 绘制引擎
const DrawEngine = {
    drawRotatedRect: function(ctx, x, y, w, h, rotation, color) {
        try {
            if (rotation !== 0) {
                ctx.save();
                const centerX = x + w / 2, centerY = y + h / 2;
                ctx.translate(centerX, centerY);
                ctx.rotate(rotation * Math.PI / 180);
                ctx.fillStyle = color;
                ctx.fillRect(-w / 2, -h / 2, w, h);
                ctx.restore();
            } else {
                ctx.fillStyle = color;
                ctx.fillRect(x, y, w, h);
            }
        } catch (error) {
            console.error("Draw rotated rect error:", error);
        }
    },
    drawGrid: function(ctx, backgroundX, backgroundY, backgroundWidth, backgroundHeight, resolutionX, resolutionY, scale) {
        try {
            ctx.beginPath();
            ctx.lineWidth = 1;
            ctx.strokeStyle = CONSTANTS.COLORS.GRID;
            for (let x = CONSTANTS.GRID_SIZE; x < resolutionX; x += CONSTANTS.GRID_SIZE) {
                const lineX = backgroundX + (x * scale);
                ctx.moveTo(lineX, backgroundY);
                ctx.lineTo(lineX, backgroundY + backgroundHeight);
            }
            for (let y = CONSTANTS.GRID_SIZE; y < resolutionY; y += CONSTANTS.GRID_SIZE) {
                const lineY = backgroundY + (y * scale);
                ctx.moveTo(backgroundX, lineY);
                ctx.lineTo(backgroundX + backgroundWidth, lineY);
            }
            ctx.stroke();
            ctx.closePath();
        } catch (error) {
            console.error("Draw grid error:", error);
        }
    }
};

function addMultiAreaConditioningCanvas(node, app) {
    const widget = {
        type: "customCanvas",
        name: "MultiAreaConditioning-Canvas",
        draw: function (ctx, node, widgetWidth, widgetY, height) {
            try {
                // 强制布局：确保控件始终位于画布下方，以覆盖ComfyUI的默认布局行为
                LayoutManager.computeCanvasSize(node);

                const visible = app.canvas && app.canvas.ds && app.canvas.ds.scale > 0.6;
                if (!visible) return;

                const margin = 10;
                const values = node.properties["values"] || [];
                const resolutionX = node.properties["width"] || CONSTANTS.DEFAULT_RESOLUTION.width;
                const resolutionY = node.properties["height"] || CONSTANTS.DEFAULT_RESOLUTION.height;
                
                if (!node.widgets || node.index === undefined || !node.widgets[node.index]) return;
                
                const curIndex = Math.round(node.widgets[node.index].value || 0);
                const canvasHeight = this.h || node.canvasHeight || CONSTANTS.CANVAS_HEIGHT;
                const availableWidth = widgetWidth - (CONSTANTS.SIDE_MARGIN * 2);
                const availableHeight = canvasHeight - (margin * 2);

                const scale = Math.min(availableWidth / resolutionX, availableHeight / resolutionY);

                const backgroundWidth = resolutionX * scale;
                const backgroundHeight = resolutionY * scale;

                // 将画布在新的可用空间内进行居中
                const backgroundX = CONSTANTS.SIDE_MARGIN + (availableWidth - backgroundWidth) / 2;
                const backgroundY = margin + (availableHeight - backgroundHeight) / 2;

                ctx.fillStyle = CONSTANTS.COLORS.BORDER;
                ctx.fillRect(backgroundX - 2, backgroundY - 2, backgroundWidth + 4, backgroundHeight + 4);
                ctx.fillStyle = CONSTANTS.COLORS.BACKGROUND;
                ctx.fillRect(backgroundX, backgroundY, backgroundWidth, backgroundHeight);
                DrawEngine.drawGrid(ctx, backgroundX, backgroundY, backgroundWidth, backgroundHeight, resolutionX, resolutionY, scale);

                for (let i = 0; i < values.length; i++) {
                    if (i === curIndex) continue;
                    const value = values[i];
                    if (!value || value.length < 4) continue;
                    let x = value[0] || 0, y = value[1] || 0, w = value[2] || 512, h = value[3] || 512, rotation = value[5] || 0.0;
                    if (x >= resolutionX || y >= resolutionY) continue;
                    if (x + w > resolutionX) w = resolutionX - x;
                    if (y + h > resolutionY) h = resolutionY - y;
                    if (w <= 0 || h <= 0) continue;
                    const color = CONSTANTS.COLORS.AREAS[i % CONSTANTS.COLORS.AREAS.length];
                    DrawEngine.drawRotatedRect(ctx, backgroundX + (x * scale), backgroundY + (y * scale), w * scale, h * scale, rotation, color);
                }

                if (curIndex < values.length) {
                    const value = values[curIndex];
                    if (value && value.length >= 4) {
                        let x = value[0] || 0, y = value[1] || 0, w = Math.max(CONSTANTS.MIN_AREA_SIZE, value[2] || 512), h = Math.max(CONSTANTS.MIN_AREA_SIZE, value[3] || 512), rotation = value[5] || 0.0;
                        if (x + w > resolutionX) w = resolutionX - x;
                        if (y + h > resolutionY) h = resolutionY - y;
                        if (w > 0 && h > 0) {
                            let areaX = backgroundX + (x * scale), areaY = backgroundY + (y * scale), areaW = w * scale, areaH = h * scale;
                            ctx.fillStyle = CONSTANTS.COLORS.SELECTED_BORDER;
                            DrawEngine.drawRotatedRect(ctx, areaX - 2, areaY - 2, areaW + 4, areaH + 4, rotation, CONSTANTS.COLORS.SELECTED_BORDER);
                            DrawEngine.drawRotatedRect(ctx, areaX, areaY, areaW, areaH, rotation, CONSTANTS.COLORS.SELECTED);
                        }
                    }
                }

                ctx.beginPath();
                ctx.fillStyle = CONSTANTS.COLORS.SELECTED;
                const slotHeight = 20;
                const TITLE_BAR_OFFSET = 0;
                const SLOT_CENTER_Y_OFFSET = 14;
                const slotY = TITLE_BAR_OFFSET + SLOT_CENTER_Y_OFFSET + (curIndex * slotHeight);
                ctx.arc(10, slotY, 5, 0, Math.PI * 2);
                ctx.fill();
                
                ctx.lineWidth = 2;
                ctx.strokeStyle = CONSTANTS.COLORS.SELECTED_BORDER;
                ctx.stroke();
                ctx.closePath();

            } catch (error) {
                console.error("Canvas draw error:", error);
            }
        }
    };
    node.addCustomWidget(widget);
    return { minWidth: 400, minHeight: 620 };
}

app.registerExtension({
    name: "Comfy.Davemane42.MultiAreaConditioning",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "MultiAreaConditioning") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const node = this;
                onNodeCreated?.apply(this, arguments);
                try {
                    if (!node.properties || !node.properties["values"]) {
                        node.properties = {
                            "values": [[0, 0, 256, 192, 1.0, 0.0], [256, 0, 256, 192, 1.0, 0.0], [0, 192, 256, 192, 1.0, 0.0], [64, 128, 128, 256, 1.0, 0.0]],
                            "width": CONSTANTS.DEFAULT_RESOLUTION.width,
                            "height": CONSTANTS.DEFAULT_RESOLUTION.height
                        };
                    }

                    addMultiAreaConditioningCanvas(node, app);
                    Utils.createCustomInt(node, "分辨率X", CONSTANTS.DEFAULT_RESOLUTION.width, function (v) { node.properties["width"] = v; });
                    Utils.createCustomInt(node, "分辨率Y", CONSTANTS.DEFAULT_RESOLUTION.height, function (v) { node.properties["height"] = v; });

                    node.index = node.widgets.length;
                    node.addWidget("slider", "当前区域索引", 3, function (v) {
                        try {
                            const selectedIndex = Math.round(v);
                            if (node.properties["values"] && selectedIndex < node.properties["values"].length) {
                                const values = node.properties["values"][selectedIndex];
                                const bottomInputs = node.widgets.slice(-6);
                                const updateIndexMap = [4, 5, 0, 1, 2, 3];
                                for (let i = 0; i < Math.min(6, bottomInputs.length); i++) {
                                    if (bottomInputs[i]) {
                                        const dataIndex = updateIndexMap[i];
                                        bottomInputs[i].value = values[dataIndex] || (dataIndex === 4 ? 1.0 : 0.0);
                                    }
                                }
                            }
                        } catch (error) { console.error("Index selection error:", error); }
                    }, { min: 0, max: 3, step: 1, precision: 0 });

                    const names = ["强度", "旋转角度", "X坐标", "Y坐标", "宽度", "高度"];
                    const defaultValues = [1.0, 0.0, 64, 128, 128, 256];
                    const paramIndexMap = [4, 5, 0, 1, 2, 3];
                    for (let i = 0; i < 6; i++) {
                        let config = {};
                        if (i === 0) config = { min: 0.0, max: 10.0, step: 0.1, precision: 2 };
                        else if (i === 1) config = { min: -180.0, max: 180.0, step: 1.0, precision: 1 };
                        Utils.createCustomInt(node, names[i], defaultValues[i], function (v) { Utils.transformFunc(this, v, node, paramIndexMap[i]); }, config);
                    }

                    node.size[0] = 400; // <--- 在这里添加，设置新节点的初始宽度
                    LayoutManager.computeCanvasSize(node);

                    let isDragging = false;
                    const onMouseDown = node.onMouseDown;
                    node.onMouseDown = function (event, pos, canvas) {
                        try {
                            if (!node.widgets || node.index === undefined || !node.widgets[node.index]) return onMouseDown?.apply(this, arguments);
                            const index = Math.round(node.widgets[node.index].value);
                            const values = node.properties["values"];
                            if (!values || index >= values.length) return onMouseDown?.apply(this, arguments);
                            
                            const resolutionX = node.properties["width"], resolutionY = node.properties["height"];
                            const canvasHeight = node.canvasHeight;
                            const margin = 10, widgetWidth = node.size[0];
                            const scale = Math.min((widgetWidth - margin * 2) / resolutionX, (canvasHeight - margin * 2) / resolutionY);
                            const backgroundWidth = resolutionX * scale, backgroundHeight = resolutionY * scale;
                            const backgroundX = margin + (widgetWidth - backgroundWidth - margin * 2) / 2, backgroundY = margin;
                            
                            const canvasWidget = this.widgets.find(w => w.type === "customCanvas");
                            const canvasOffsetY = canvasWidget ? canvasWidget.y : CONSTANTS.TITLE_HEIGHT;
                            const relativeX = pos[0] - backgroundX, relativeY = pos[1] - backgroundY - canvasOffsetY;

                            if (relativeX >= 0 && relativeX <= backgroundWidth && relativeY >= 0 && relativeY <= backgroundHeight) {
                                let areaX = values[index][0], areaY = values[index][1], areaW = values[index][2], areaH = values[index][3];
                                const screenAreaX = backgroundX + (areaX * scale), screenAreaY = backgroundY + (areaY * scale), screenAreaW = areaW * scale, screenAreaH = areaH * scale;
                                const mouseX = pos[0], mouseY = pos[1] - canvasOffsetY;
                                if (isDragging) {
                                    isDragging = false;
                                    canvas.style.cursor = 'default';
                                    return true;
                                }
                                if (mouseX >= screenAreaX && mouseX <= screenAreaX + screenAreaW && mouseY >= screenAreaY && mouseY <= screenAreaY + screenAreaH) {
                                    isDragging = true;
                                    canvas.style.cursor = 'move';
                                } else {
                                    const x = Math.round((relativeX / backgroundWidth) * resolutionX), y = Math.round((relativeY / backgroundHeight) * resolutionY);
                                    const clampedX = Math.max(0, Math.min(resolutionX - areaW, x)), clampedY = Math.max(0, Math.min(resolutionY - areaH, y));
                                    values[index][0] = clampedX;
                                    values[index][1] = clampedY;
                                    const xWidget = node.widgets.find(w => w.name === "x"), yWidget = node.widgets.find(w => w.name === "y");
                                    if (xWidget) xWidget.value = clampedX;
                                    if (yWidget) yWidget.value = clampedY;
                                }
                                return true;
                            }
                            return onMouseDown?.apply(this, arguments);
                        } catch (error) {
                            console.error("Mouse down error:", error);
                            return onMouseDown?.apply(this, arguments);
                        }
                    };
                    
                    const onMouseMove = node.onMouseMove;
                    node.onMouseMove = function (event, pos, canvas) {
                        try {
                            if (isDragging) {
                                const index = Math.round(node.widgets[node.index].value);
                                const values = node.properties["values"];
                                const resolutionX = node.properties["width"], resolutionY = node.properties["height"];
                                const canvasHeight = node.canvasHeight;
                                const margin = 10, widgetWidth = node.size[0];
                                const scale = Math.min((widgetWidth - margin * 2) / resolutionX, (canvasHeight - margin * 2) / resolutionY);
                                const backgroundWidth = resolutionX * scale, backgroundHeight = resolutionY * scale;
                                const backgroundX = margin + (widgetWidth - backgroundWidth - margin * 2) / 2, backgroundY = margin;
                                
                                const canvasWidget = this.widgets.find(w => w.type === "customCanvas");
                                const canvasOffsetY = canvasWidget ? canvasWidget.y : CONSTANTS.TITLE_HEIGHT;
                                const relativeX = pos[0] - backgroundX, relativeY = pos[1] - backgroundY - canvasOffsetY;
                                const mouseRealX = (relativeX / backgroundWidth) * resolutionX, mouseRealY = (relativeY / backgroundHeight) * resolutionY;
                                const currentW = values[index][2] || 512, currentH = values[index][3] || 512;
                                let newX = mouseRealX - currentW / 2, newY = mouseRealY - currentH / 2;
                                newX = Math.max(0, Math.min(resolutionX - currentW, newX));
                                newY = Math.max(0, Math.min(resolutionY - currentH, newY));
                                values[index][0] = Math.round(newX);
                                values[index][1] = Math.round(newY);
                                const xWidget = node.widgets.find(w => w.name === "x"), yWidget = node.widgets.find(w => w.name === "y");
                                if (xWidget) xWidget.value = Math.round(newX);
                                if (yWidget) yWidget.value = Math.round(newY);
                                return true;
                            }
                            return onMouseMove?.apply(this, arguments);
                        } catch (error) {
                            console.error("Mouse move error:", error);
                            return onMouseMove?.apply(this, arguments);
                        }
                    };
                } catch (error) {
                    console.error("Node creation error:", error);
                }
            };
            
            const onLoadedGraphNode = nodeType.prototype.onLoadedGraphNode;
            nodeType.prototype.onLoadedGraphNode = function(nodeData, app) {
                try {
                    onLoadedGraphNode?.apply(this, arguments);
                    if (!this.properties) this.properties = {};
                    if (!this.properties["values"]) this.properties["values"] = [[0, 0, 256, 192, 1.0, 0.0], [256, 0, 256, 192, 1.0, 0.0], [0, 192, 256, 192, 1.0, 0.0], [64, 128, 128, 256, 1.0, 0.0]];
                    if (!this.properties["width"]) this.properties["width"] = CONSTANTS.DEFAULT_RESOLUTION.width;
                    if (!this.properties["height"]) this.properties["height"] = CONSTANTS.DEFAULT_RESOLUTION.height;
                    
                    // 如果保存的节点宽度不存在或小于400，则设置为400
                    if (!this.size[0] || this.size[0] < 400) {
                        this.size[0] = 400; // <--- 在这里添加
                    }
                    LayoutManager.computeCanvasSize(this);

                } catch (error) {
                    console.error("Load graph node error:", error);
                }
            };
        }
    }
});