import { app } from "../../../scripts/app.js";
import { mkName } from "../utils.js";

const PACKAGE_NAME = "Tagger";
const CLASS_NAMES = [mkName(PACKAGE_NAME, "Tagger")];

const WIDGET_ORIGINALS = new WeakMap();

const THRESHOLD_WIDGETS = [
    "threshold",
    "character_threshold",
    "copyright_threshold",
    "artist_threshold",
    "meta_threshold",
    "rating_threshold",
];

const INCLUDE_THRESHOLD_PAIRS = [
    ["include_general", "threshold"],
    ["include_character", "character_threshold"],
    ["include_copyright", "copyright_threshold"],
    ["include_artist", "artist_threshold"],
    ["include_meta", "meta_threshold"],
    ["include_rating", "rating_threshold"],
];

function findWidget(node, name) {
    return node.widgets?.find(w => w.name === name);
}

function rememberWidget(widget) {
    if (!widget || WIDGET_ORIGINALS.has(widget)) return;
    WIDGET_ORIGINALS.set(widget, {
        type: widget.type,
        computeSize: widget.computeSize,
        optionsHidden: widget.options?.hidden,
    });
}

function setWidgetVisible(node, name, visible) {
    const widget = findWidget(node, name);
    if (!widget) return;

    rememberWidget(widget);
    const original = WIDGET_ORIGINALS.get(widget);

    widget.options ??= {};
    if (visible) {
        widget.type = original.type;
        widget.computeSize = original.computeSize;
        widget.hidden = false;
        widget.disabled = false;
        widget.options.hidden = original.optionsHidden ?? false;
    } else {
        widget.type = original.type;
        widget.computeSize = () => [0, -4];
        widget.hidden = true;
        widget.disabled = true;
        widget.options.hidden = true;
    }

    widget.linkedWidgets?.forEach(linkedWidget => {
        rememberWidget(linkedWidget);
        const linkedOriginal = WIDGET_ORIGINALS.get(linkedWidget);
        linkedWidget.options ??= {};
        if (visible) {
            linkedWidget.type = linkedOriginal.type;
            linkedWidget.computeSize = linkedOriginal.computeSize;
            linkedWidget.hidden = false;
            linkedWidget.disabled = false;
            linkedWidget.options.hidden = linkedOriginal.optionsHidden ?? false;
        } else {
            linkedWidget.type = linkedOriginal.type;
            linkedWidget.computeSize = () => [0, -4];
            linkedWidget.hidden = true;
            linkedWidget.disabled = true;
            linkedWidget.options.hidden = true;
        }
    });
}

function refreshNodeWidgets(node) {
    if (!node.widgets) return;
    node.widgets = [...node.widgets];
}

function resizeNodeToWidgets(node) {
    const computedSize = node.computeSize?.();
    if (computedSize && node.size) {
        node.size[1] = computedSize[1];
    }
}

function updateThresholdWidgets(node) {
    const useDefaultThreshold = Boolean(findWidget(node, "use_default_threshold")?.value);
    const mcutThreshold = Boolean(findWidget(node, "mcut_threshold")?.value);
    const thresholdInputsEnabled = !useDefaultThreshold && !mcutThreshold;

    for (const [includeName, thresholdName] of INCLUDE_THRESHOLD_PAIRS) {
        const includeWidget = findWidget(node, includeName);
        const visible = thresholdInputsEnabled && Boolean(includeWidget?.value);
        setWidgetVisible(node, thresholdName, visible);
    }

    refreshNodeWidgets(node);
    resizeNodeToWidgets(node);
    app.graph?.setDirtyCanvas(true, true);
}

function interceptWidgetValue(widget, onChange) {
    if (!widget || widget._jupoTaggerVisibilityIntercepted) return;
    widget._jupoTaggerVisibilityIntercepted = true;

    const descriptor =
        Object.getOwnPropertyDescriptor(widget, "value") ||
        Object.getOwnPropertyDescriptor(Object.getPrototypeOf(widget), "value");
    let widgetValue = widget.value;

    Object.defineProperty(widget, "value", {
        configurable: true,
        enumerable: true,
        get() {
            return descriptor?.get
                ? descriptor.get.call(widget)
                : widgetValue;
        },
        set(newValue) {
            if (descriptor?.set) {
                descriptor.set.call(widget, newValue);
            } else {
                widgetValue = newValue;
            }
            onChange(newValue);
        },
    });
}

function setupThresholdWidgetVisibility(node) {
    THRESHOLD_WIDGETS.forEach(name => rememberWidget(findWidget(node, name)));
    interceptWidgetValue(findWidget(node, "use_default_threshold"), () => updateThresholdWidgets(node));
    interceptWidgetValue(findWidget(node, "mcut_threshold"), () => updateThresholdWidgets(node));
    INCLUDE_THRESHOLD_PAIRS.forEach(([includeName]) => {
        interceptWidgetValue(findWidget(node, includeName), () => updateThresholdWidgets(node));
    });
    updateThresholdWidgets(node);
}

const extension = {
    name: mkName(PACKAGE_NAME, "Tagger"), 

    beforeRegisterNodeDef: async function(nodeType, nodeData, app) {
        if (!CLASS_NAMES.includes(nodeType.comfyClass)) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const result = onNodeCreated?.apply(this, arguments);
            setupThresholdWidgetVisibility(this);
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function() {
            const result = onConfigure?.apply(this, arguments);
            updateThresholdWidgets(this);
            return result;
        };
    },
};

app.registerExtension(extension);

