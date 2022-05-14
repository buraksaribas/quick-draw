const paintCanvas = document.querySelector( '.paint-canvas' );
const context = paintCanvas.getContext( '2d' );
context.lineCap = 'round';

const colorPicker = document.querySelector( '.color-picker');

colorPicker.addEventListener( 'change', event => {
    context.strokeStyle = event.target.value; 
} );

const lineWidthRange = document.querySelector( '.line-range' );
const lineWidthLabel = document.querySelector( '.range-value' );

lineWidthRange.addEventListener( 'input', event => {
    const width = event.target.value;
    lineWidthLabel.innerHTML = width;
    context.lineWidth = width;
} );

let x = 0, y = 0;
let isMouseDown = false;

const stopDrawing = () => { isMouseDown = false; }

const startDrawing = event => {
    isMouseDown = true;   
   [x, y] = [event.offsetX, event.offsetY];  
}
const drawLine = event => {
    if ( isMouseDown ) {
        const newX = event.offsetX;
        const newY = event.offsetY;
        context.beginPath();
        context.moveTo( x, y );
        context.lineTo( newX, newY );
        context.stroke();
        
        x = newX;
        y = newY;
    }
}

const clearDrawing = event => {
    const displayconv = paintCanvas.getContext( 'displayarea' );
    context.clearRect(0, 0, paintCanvas.width, paintCanvas.height);
    displayconv.clearRect(0, 0, paintCanvas.width, paintCanvas.height);
}

const submitDrawing = event => {

}

paintCanvas.addEventListener( 'mousedown', startDrawing );
paintCanvas.addEventListener( 'mousemove', drawLine );
paintCanvas.addEventListener( 'mouseup', stopDrawing );
paintCanvas.addEventListener( 'mouseout', stopDrawing );