
$('#input_url_button').click(function() {
	load_image_from_url($('#input_image_url').val());
});

$('#input_image_file').change(function() {
	load_image_from_file(this);
});

$('#input_select_full_button').click(function() {
	var image_node = $('#image').get(0);
	if(area_select) {
		area_select.setSelection(0, 0, 
			image_node.width,
			image_node.height);
		area_select.setOptions({ 
			imageWidth: image_node.width,
			imageHeight: image_node.height,
			show: true
		});
		area_select.update();
		display_crop(image_node, area_select.getSelection());
		update_crop(image_node, area_select.getSelection());
	}
});

var area_select = null;
$('#image')
	.on('load', function() {
		area_select = $(this).imgAreaSelect({
			imageWidth: $(this).width(),
			imageHeight: $(this).height(),
			handles: true,
			onSelectChange: display_crop,
			onSelectEnd: update_crop,
			instance: true
		});
		$(this).show();
		$('#block_crop_selection_right').show();
	})
	.hide();

$('#form_search').submit(function(e) {
	e.preventDefault();
	search_image();
	return false;
});

$('#block_crop_selection_right').hide();

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
}

function update_crop(img, selection) {
	var image_scale_x = img.naturalWidth / img.width;
	var image_scale_y = img.naturalHeight / img.height;
	$('#input_crop_x1').val(Math.round(selection.x1 * image_scale_x));
	$('#input_crop_y1').val(Math.round(selection.y1 * image_scale_y));
	$('#input_crop_x2').val(Math.round(selection.x2 * image_scale_x));
	$('#input_crop_y2').val(Math.round(selection.y2 * image_scale_y));
}

function load_image_from_url(url) {
	if(!url || url === "") {
		return;
	}

	$('#image').hide();
	$('#input_search_url').val("");

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

	var reader = new FileReader();
	$(reader).on('load', function(e) {
		$('#block_crop_selection_right').hide();
		$('#image').attr('src', e.target.result);
		$('#crop').hide().attr('src', e.target.result);
		$('#input_search_file').attr('files', file_node.files);
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