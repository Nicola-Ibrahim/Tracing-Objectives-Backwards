declare module 'react-plotly.js' {
    import * as React from 'react';
    import { PlotData, Layout, Config } from 'plotly.js';

    export interface PlotParams {
        data: Partial<PlotData>[];
        layout: Partial<Layout>;
        config?: Partial<Config>;
        style?: React.CSSProperties;
        useResizeHandler?: boolean;
        onInitialized?: (figure: Readonly<Figure>, graphDiv: Readonly<HTMLElement>) => void;
        onUpdate?: (figure: Readonly<Figure>, graphDiv: Readonly<HTMLElement>) => void;
        onPurge?: (figure: Readonly<Figure>, graphDiv: Readonly<HTMLElement>) => void;
        onError?: (err: Readonly<Error>) => void;
        className?: string;
    }

    export interface Figure {
        data: PlotData[];
        layout: Layout;
    }

    export default class Plot extends React.Component<PlotParams> {}
}
