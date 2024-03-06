/**
   TODO:
     [1] Implement detection against noise
     [2] Find anchors
     [3] Gap algorithm
**/

function euclid(x, y) {
    let n = x.length;
    let distance = 0.0;
    for(let i = 0; i < n; i++) {
	distance += Math.pow(x[i] - y[i], 2);
    }
    return Math.sqrt(distance);
}

function min3(x, y, z) {
    let min = x;
    if(y < min) min = y;
    if(z < min) min = z;
    return min;
}

function dtw(x, y) {
    let n = x.length;
    let m = y.length;
    let d = x[0].length;

    let dp = 0; // TODO this is harder than i want to
    dp[0][0] = 0.0; 
    for(let i = 1; i < n + 1; i++) {
	for(let j = 1; j < m + 1; j++) {
	    dp[i][j] = min3(
		dp[i - 1][j],
		dp[i][j - 1],
		dp[i-1][j - 1]
	    );
	    dp[i][j] += euclid(x[i - 1], y[j - 1]);
	}
    }
    return dp[n][m];    
}

function largest_width(images) {
    let maxLen = 0;
    for(let image of images) {
	let id = image.id;
	let gaps = get_gaps(id);
	let w = image.width;
	for(let gap of gaps) {
	    w += gap.len;
	}
	maxLen = Math.max(w, maxLen);
    }
    return maxLen;
}

function get_gaps(id) {
    // TODO mock
    gaps = {
	"05291001_RegPCA110_3969000_6615000": [
	    {start: 500, len: 1000},
	    {start: 2500, len: 4000},
	    {start: 12500, len: 2000},
	]
    };
    
    return gaps[id] ?? [];
}

function getMousePos(canvas, evt) {
    var rect = canvas.getBoundingClientRect();
    return {
        x: evt.clientX - rect.left,
        y: evt.clientY - rect.top
    };
}

function spectrograms(images, canvas) {
    const max_width = largest_width(images);
    canvas.width = max_width;
    canvas.height = images.length * images[0].height;

    canvas.addEventListener("click", function (evt) {
	var mousePos = getMousePos(canvas, evt);
	console.log(mousePos.x + ',' + mousePos.y);
    }, false);

    // TODO pull this out
    const context = canvas.getContext('2d');    
    for(let i = 0; i < images.length; i++) {
	let id = images[i].id;
	let gaps = get_gaps(id);

	let cur_img = 0;
	let cum_gap = 0;
	for(let gap of gaps) {
	    let cur_canvas = cur_img + cum_gap;
	    console.log(cur_img, gap.start - cur_img, cur_canvas);
	    context.drawImage(images[i],
			      cur_img, 0, gap.start - cur_img, images[i].height,
			      cur_canvas, i * images[i].height, gap.start - cur_img, images[i].height);
	    context.fillStyle = "red";
	    context.fillText(cur_canvas, cur_canvas, i * images[i].height + 10);
	    context.fillText(cur_canvas + (gap.start - cur_img), cur_canvas +  (gap.start - cur_img), i * images[i].height + 10);
	    cur_img = gap.start;
	    cum_gap += gap.len;
	}
	let cur_canvas = cur_img + cum_gap;
	context.fillText(cur_canvas, cur_canvas, i * images[i].height);
	context.drawImage(images[i],
			  cur_img, 0, images[i].width - cur_img, images[i].height,
			  cur_canvas, i * images[i].height, images[i].width - cur_img, images[i].height);		
	context.fillStyle = "red";
	context.fillText(cur_canvas, cur_canvas, i * images[i].height + 10);
	context.fillText(cur_canvas + (images[i].width - cur_img), cur_canvas + (images[i].width - cur_img) - 30, i * images[i].height + 10);
	console.log('---');
    }
}

function getData(url) {
    return fetch(url)
	.then(response=>{
	    return response.json()
	})
}
