const hsvToRgb = (h, s, v) => {
    h = h % 360;
    s = Math.max(0, Math.min(1, s));
    v = Math.max(0, Math.min(1, v));
    
    const c = v * s;
    const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
    const m = v - c;

    let r = 0, g = 0, b = 0;
    
    if (h < 60) {
        [r, g, b] = [c, x, 0];
    } else if (h < 120) {
        [r, g, b] = [x, c, 0];
    } else if (h < 180) {
        [r, g, b] = [0, c, x];
    } else if (h < 240) {
        [r, g, b] = [0, x, c];
    } else if (h < 300) {
        [r, g, b] = [x, 0, c];
    } else {
        [r, g, b] = [c, 0, x];
    }

    return {
        r: Math.round((r + m) * 255),
        g: Math.round((g + m) * 255),
        b: Math.round((b + m) * 255)
    };
};

const hsvToRgbaString = (h, s, v, a = 1) => {
    const rgb = hsvToRgb(h, s, v);
    return `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${a})`;
};

const rgbToHsv = (r, g, b) => {
    r /= 255;
    g /= 255;
    b /= 255;

    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const diff = max - min;

    let h = 0;
    const s = max === 0 ? 0 : diff / max;
    const v = max;

    if (diff !== 0) {
        switch (max) {
            case r:
                h = 60 * ((g - b) / diff + (g < b ? 6 : 0));
                break;
            case g:
                h = 60 * ((b - r) / diff + 2);
                break;
            case b:
                h = 60 * ((r - g) / diff + 4);
                break;
        }
    }

    return { h, s, v };
};

const getVisitedColor = (visitCount) => {
    const baseColor = "#FFD700";
    const alpha = Math.min(0.2 + (visitCount * 0.15), 1);
    return `${baseColor}${Math.round(alpha * 255).toString(16).padStart(2, "0")}`;
};

export { hsvToRgb, hsvToRgbaString, rgbToHsv, getVisitedColor };