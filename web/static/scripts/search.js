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
	if(img.naturalWidth < img.naturalHeight) {
		$(this).css({'min-width': '', 'max-width': '', 'flex-basis': '', 
					 'min-height': '100%', 'max-height': '100%'});
	} else {
		$(this).css({'min-height': '', 'max-height': '', 'flex-basis': '100%', 
					 'min-width': '100%', 'max-width': '100%'});
	}
	$('#text_image_size').text("WxH: " + 
		this.naturalWidth + "x" + this.naturalHeight
	);
	$(this).show();
	$('#section_crop_selection').show();
	$('#block_crop_selection_right').show();
})
.hide();

$('#input_url_button').click(function() {
	load_image_from_url($('#input_image_url').val());
});

$('#input_image_file').change(function() {
	load_image_from_file(this);
});

$('#mode_rerank').click(function() {
	if($(this).prop('checked')) {
		$('#mode_localization').prop('checked', true);
	}
});

$('#mode_localization').click(function(e) {
	if($('#mode_rerank').prop('checked')) {
		e.preventDefault();
		return false;
	}
});

$('#form_search').submit(function(e) {
	e.preventDefault();
	search_image();
	return false;
});

$('#block_selection_image').click(function(event) {
	if(area_select) {
		area_select.cancelSelection();
	}
});

// Make canvas absorb the close onclick event from the lightbox
$('#canvas').click(function(event) {
	event.stopPropagation();
});

$('#section_crop_selection').hide();
$('#section_results').hide();

function compute_bounding_box(selection) {
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

	var bounding_box = compute_bounding_box(selection);
	$('#text_crop_params').text(
		'x: ' + bounding_box[0] + ':' + bounding_box[2] + ', ' +
		'y: ' + bounding_box[1] + ':' + bounding_box[3]
	);
}

function update_crop(img, selection) {
	var bounding_box = compute_bounding_box(selection);
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
	$('#section_results').hide();
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
	$('#section_results').hide();
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

function disable_search_button() {
	$('#search_submit').attr('disabled', true);
	$('#search_submit_label').addClass('button_disabled');
}

function enable_search_button() {
	$('#search_submit_label').removeClass('button_disabled');
	$('#search_submit').attr('disabled', false);
}

function search_image() {
	disable_search_button();

	var url = '';
	var fd = new FormData($('#form_search').get(0));
	if(!fd.get('url') || fd.get('url') === "") {
		fd.delete('url');
	}

	var file_node = $('#input_image_file').get(0);
	if(file_node.files && file_node.files[0]) {
		fd.append('file', file_node.files[0]);
		url = 'search_file';
	} else {
		url = 'search_url';
	}

	if(!fd.has('file') && !fd.has('url')) {
		return;
	}

	var selection = area_select.getSelection();
	console.log(selection);
	
	var bounding_box = compute_bounding_box(selection);
	fd.set('x1', bounding_box[0]);
	fd.set('y1', bounding_box[1]);
	fd.set('x2', bounding_box[2]);
	fd.set('y2', bounding_box[3]);

	$('#block_results_images').hide();
	$('#block_results_status').empty();
	$('#block_results_spinner').show();
	$('#section_results').show();

	var start_time = new Date().getTime();
	$.ajax({
		url: url,
		type: 'POST',
		dataType: 'json',
		data: fd,
		processData: false,
		contentType: false
	}).done(function(response, xhr) {
		var request_time = (new Date().getTime() - start_time) / 1000;
		var status = 'Retrieved ' + response['results'].length + ' results in ' + request_time + ' seconds.';
		$('#block_results_status').html(status);
		construct_results(response['results']);
		$('#block_results_spinner').hide();
		$('#block_results_images').show();
		enable_search_button();
	}).fail(function(xhr, status, error) {
		console.log(status);
		console.log(error);
		var status = '<span class="important">An error occured while searching: ' + xhr.responseJSON['message'] + '</span>';
		$('#block_results_status').html(status);
		$('#block_results_spinner').hide();
		enable_search_button();
	});
}

function construct_results(images) {
	$('#block_results_images_left').empty().append('<hr class="hr_sep"/>');
	$('#block_results_images_right').empty().append('<hr class="hr_sep"/>');
	var idx = 0;
	for(image of images) {
		var bbox = '0, 0, 0, 0';
		if(image.bbox) {
			bbox = image.bbox.x1 + ', ' + image.bbox.y1 + ', ' + image.bbox.x2 + ', ' + image.bbox.y2;
		}

		var div = '<div class="block_result"><div class="block_result_image">';
		if(image.url) {
			div += '<img id="result' + idx + '" src="/' + image.url 
				+ '" class="result_image" onclick="open_lightbox(result'
				+ idx + ', ' + bbox + ')"></div>';
		} else {
			div += '<span class="fa fa-picture-o fa-2x result_image"></span></div>';
		}
		div += '<div class="block_result_info"><ul><li>Name: ' + image.name
			+ '</li><li>Score: ' + image.score + '</li>';
		if(image.bbox) {
			div += '<li>Box: ' + image.bbox.x1 + ":" + image.bbox.x2 + ", " 
				+ image.bbox.y1 + ":" + image.bbox.y2 + '</li>';
		}
		if(image.ext_url) {
			div += '<li><a href="' + image.ext_url + '">External link</a></li>';
		}
		div += '</ul></div></div><hr class="hr_sep"/>';
		if(idx % 2 == 0) {
			$('#block_results_images_left').append(div);
		} else {
			$('#block_results_images_right').append(div);
		}
		idx += 1;
	}
}

function open_lightbox(img, x1, y1, x2, y2) {
	$('#block_lightbox').show();
	var canvas = $('#canvas').get(0)
	var ctx = canvas.getContext('2d');

	// Scale s.t. the larger side gets fitted exactly onto the canvas
	var ratio  = Math.min(canvas.width / img.naturalWidth, 
						  canvas.height / img.naturalHeight);
	// Shift s.t. the image is centered
	var shift_x = Math.round((canvas.width - img.naturalWidth * ratio) / 2);
	var shift_y = Math.round((canvas.height - img.naturalHeight * ratio) / 2);

	ctx.clearRect(0, 0, canvas.width, canvas.height);
	ctx.drawImage(img, shift_x, shift_y, 
				  Math.round(img.naturalWidth * ratio), 
				  Math.round(img.naturalHeight * ratio));
	ctx.beginPath();
	ctx.lineWidth = "2";
	ctx.strokeStyle = "red";
	ctx.rect(Math.round(x1*ratio)+shift_x, Math.round(y1*ratio)+shift_y,
			 Math.round((x2-x1)*ratio), Math.round((y2-y1)*ratio));
	ctx.stroke();
}

function close_lightbox() {
	$('#block_lightbox').hide();
}
