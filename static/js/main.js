function showMyImage(fileInput) {
    var files = fileInput.files;
      //console.log(files);
      for (var i = 0; i < files.length; i++) {           
          var file = files[i];
          console.log(file.name);
          var imageType = /image.*/;     
          if (!file.type.match(imageType)) {
              continue;
          }           
          var img=document.getElementById("thumbnil");            
          img.file = file;    
          var reader = new FileReader();
          reader.onload = (function(aImg) { 
              return function(e) { 
                  aImg.src = e.target.result; 
              }; 
          })(img);
          reader.readAsDataURL(file);
          thumbnil.style.display = 'block';
          //$("#banner_name").text(file.name);

      }
}