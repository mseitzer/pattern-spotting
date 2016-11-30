var area_select = null;

var image_scale_x, image_scale_y;

$(document).ready(function() {
	area_select = $('#image').imgAreaSelect({
		handles: true,
		onSelectChange: display_crop,
		instance: true,
	});
});

$('#image').on('load', function() {
	var img = this;
	setTimeout(function() {
		/* This has to be put into a timeout, otherwise the correct
		 * width and height is not yet accessible.
		 */
		image_scale_x = img.naturalWidth / img.width;
		image_scale_y = img.naturalHeight / img.height;
		area_select.setOptions({
			minWidth: Math.round(32 / image_scale_x) || 1,
			minHeight: Math.round(32 / image_scale_y) || 1,
		});
	}, 0);

	$('#text_image_size').text(
		this.naturalWidth + "px, " + this.naturalHeight + "px"
	);
	$(this).show();
	$('#block_crop_selection').show();
	$('#block_crop_selection_right').show();
})
.hide();

$('#input_url_button').click(function() {
	load_image_from_url($('#input_image_url').val());
});

$('#input_image_file').change(function() {
	load_image_from_file(this);
});

$('#form_search').submit(function(e) {
	e.preventDefault();
	search_image();
	return false;
});

$('#block_crop_selection').hide();

function compute_bounding_box(img, selection) {
	return [
		Math.round(selection.x1 * image_scale_x),
		Math.round(selection.y1 * image_scale_y),
		Math.round(selection.x2 * image_scale_x),
		Math.round(selection.y2 * image_scale_y)
	];
}

function display_crop(img, selection) {
	var crop_block_node = $('#block_crop_image');
	var crop_scale_x = crop_block_node.width() / (selection.width || 1);
	var crop_scale_y = crop_block_node.height() / (selection.height || 1);
	$('#crop').css({
		width: Math.round(crop_scale_x * img.width) + 'px',
		height: Math.round(crop_scale_y * img.height) + 'px',
		marginLeft: '-' + Math.round(crop_scale_x * selection.x1) + 'px',
		marginTop: '-' + Math.round(crop_scale_y * selection.y1) + 'px'
	}).show();

	var bounding_box = compute_bounding_box(img, selection);
	$('#text_crop_params').text(
		  bounding_box[0] + ":" + bounding_box[2] + ", " 
		+ bounding_box[1] + ":" + bounding_box[3]
	);
}

function update_crop(img, selection) {
	var bounding_box = compute_bounding_box(img, selection);
	$('#input_crop_x1').val(bounding_box[0]);
	$('#input_crop_y1').val(bounding_box[1]);
	$('#input_crop_x2').val(bounding_box[2]);
	$('#input_crop_y2').val(bounding_box[3]);
}

function load_image_from_url(url) {
	if(!url || url === "") {
		return;
	}

	$('#image').hide();
	$('#input_search_url').val("");
	$('#input_image_file').val("");
	if(area_select) {
		area_select.setOptions({hide: true});
		area_select.update();
	}

	var img = new Image();
	img.onload = function() {
		$('#block_crop_selection_right').hide();
		$('#image').attr('src', img.src);
		$('#crop').hide().attr('src', img.src);
		$('#input_search_url').val(url);
	}

	img.src = url;
}

function load_image_from_file(file_node) {
	if(!file_node.files && !file_node.files[0]) {
		return;
	}

	// TODO: file name extension validation

	$('#image').hide();
	$('#input_search_url').val("");
	if(area_select) {
		area_select.setOptions({hide: true});
		area_select.update();
	}

	var reader = new FileReader();
	$(reader).on('load', function(e) {
		$('#block_crop_selection_right').hide();
		$('#image').attr('src', e.target.result);
		$('#crop').hide().attr('src', e.target.result);
	})
	reader.readAsDataURL(file_node.files[0]);
}

function search_image() {
	var fd = new FormData($('#form_search').get(0));
	if(!fd.get('url') || fd.get('url') === "") {
		fd.delete('url');
	}

	var file_node = $('#input_image_file').get(0);
	if(file_node.files && file_node.files[0]) {
		fd.append('file', file_node.files[0]);
	}

	if(!fd.has('file') && !fd.has('url')) {
		return;
	}

	var selection = area_select.getSelection();
	update_crop($('#image').get(0), selection);

	if(fd.get('x1') === "" 
		|| fd.get('y1') === "" 
		|| fd.get('x2') === "" 
		|| fd.get('y2') === "") {
		return;
	}

	$.ajax({
		url: 'search',
		type: 'POST',
		data: fd,
		processData: false,
		contentType: false
	}).done(function(response, xhr) {
		console.log(response);
	}).fail(function(xhr, status, error) {
		// TODO: error handling
		console.log(status);
		console.log(error);
	});
}