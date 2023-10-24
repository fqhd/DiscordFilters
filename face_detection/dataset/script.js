let selectionSquareWidth = 128;

let current_image_idx = 0;
let current_image;
let csv_string = '';
let paragraph;

function preload() {
	current_image = loadImage('frames/0.jpg');
}

function setup() {
	createCanvas(640, 360);
	background(0);
	textSize(50);
}

function draw() {
	// Only draw the selection rectangle here
	clear();
	image(current_image, 0, 0, 640, 360);
	fill(10, 120, 220, 50);
	square(mouseX - selectionSquareWidth / 2, mouseY - selectionSquareWidth / 2, selectionSquareWidth);
	fill(0, 0, 0, 255);
	text(`${current_image_idx}`, 5, 45);
}

function tick() {
	if(current_image_idx == 500) {
		if(!paragraph) {
			paragraph = createP(csv_string);
		}
	}else{
		// Save the image and the location of the cursor in a csv
		const normalized_x_coord = (mouseX - selectionSquareWidth / 2) / 640;
		const normalized_y_coord = (mouseY - selectionSquareWidth / 2) / 360;
		const normalized_selection_square_width = (selectionSquareWidth - 80) / (392 - 80);
		csv_string += `frames/${current_image_idx}.jpg,${normalized_x_coord},${normalized_y_coord},${normalized_selection_square_width}<br/>`;

		current_image_idx++;
		loadImage(`frames/${current_image_idx}.jpg`, img => {
			current_image = img;
		});
	}
}

function mouseClicked() {
	tick();
}

function mouseWheel(event) {
	if(event.delta > 0) {
		selectionSquareWidth += 12;
	}else{
		selectionSquareWidth -= 12;
	}
	if(selectionSquareWidth > 392) {
		selectionSquareWidth = 392;
	}
	if(selectionSquareWidth < 80) {
		selectionSquareWidth = 80;
	}
}