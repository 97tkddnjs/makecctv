var pw1 = document.querySelector('#password');
var pwMsg = document.querySelector('#alertTxt');
var pwImg1 = document.querySelector('#pswd1_img1');

var pw2 = document.querySelector('#confirm');
var pwImg2 = document.querySelector('#pswd2_img1');
var pwMsgArea = document.querySelector('.idForm');

var userName = document.querySelector('#username');

var FirstName = document.querySelector('#firstname')
var LastName = document.querySelector('#lastname')

var Email = document.querySelector('#email');

var error = document.querySelectorAll('.error_next_box');


/*이벤트 핸들러 연결*/

pw1.addEventListener("focusout", checkPw);
pw2.addEventListener("focusout", comparePw);
userName.addEventListener("focusout", checkName);
Email.addEventListener("focusout", isEmailCorrect);
FirstName.addEventListener("focusout", checkLastName);
LastName.addEventListener("focusout", checkFirstName);


/*콜백 함수*/

/* ID */
function checkName() {
    var idPattern = /[a-zA-Z0-9_-]{5,20}/;
    if (userName.value === "") {
        error[0].innerHTML = "필수 정보입니다.";
        error[0].style.display = "block";
    } else if (!idPattern.test(userName.value)) {
        error[0].innerHTML = "5~20자의 영문 대 소문자, 숫자만 사용 가능합니다.";
        error[0].style.display = "block";
    } else {
        error[0].innerHTML = "사용 가능한 아이디입니다!";
        error[0].style.color = "#08A600";
        error[0].style.display = "block";
    }
}

/* 비밀번호 */
function checkPw() {
    var pwPattern = /[a-zA-Z0-9~!@#$%^&*()_+|<>?:{}]{8,16}/;
    if (pw1.value === "") {
        error[1].innerHTML = "필수 정보입니다.";
        error[1].style.display = "block";
    } else if (!pwPattern.test(pw1.value)) {
        error[1].innerHTML = "8~16자 영문 대 소문자, 숫자, 특수문자를 사용하세요.";
        pwMsg.innerHTML = "사용불가";
        pwMsgArea.style.paddingRight = "93px";
        error[1].style.display = "block";

        pwMsg.style.display = "block";
        pwImg1.src = "/static/img/m_icon_not_use.png";
    } else {
        error[1].style.display = "none";
        pwMsg.innerHTML = "안전";
        pwMsg.style.display = "block";
        pwMsg.style.color = "#03c75a";
        pwImg1.src = "/static/img/m_icon_safe.png";
    }
}

/* 비밀번호 재확인 */
function comparePw() {
    if (pw2.value === pw1.value && pw2.value !== "") {
        pwImg2.src = "/static/img/m_icon_check_enable.png";
        error[2].style.display = "none";
    } else if (pw2.value !== pw1.value) {
        pwImg2.src = "/static/img/m_icon_check_disable.png";
        error[2].innerHTML = "비밀번호가 일치하지 않습니다.";
        error[2].style.display = "block";
    }

    if (pw2.value === "") {
        error[2].innerHTML = "필수 정보입니다.";
        error[2].style.display = "block";
    }
}

/* 이메일 */
function isEmailCorrect() {
    var emailPattern = /[a-z0-9]{2,}@[a-z0-9-]{2,}\.[a-z0-9]{2,}/;
    if (Email.value === "") {
        error[3].innerHTML = "필수 정보입니다.";
        error[3].style.display = "block";
    } else if (!emailPattern.test(Email.value)) {
        error[3].innerHTML = "이메일 형식이 아닙니다."
        error[3].style.display = "block";
    } else {
        error[3].style.display = "none";
    }

}

/* 이름 */
function checkLastName() {
    var namePattern = /[a-zA-Z가-힣]/;
    if (LastName.value === "") {
        error[4].innerHTML = "필수 정보입니다.";
        error[4].style.display = "block";
    } else if (!namePattern.test(LastName.value)) {
        error[4].innerHTML = "한글과 영문 대 소문자를 사용하세요. (특수기호, 공백 사용 불가)";
        error[4].style.display = "block";
    } else {
        error[4].style.display = "none";
    }
}

/* 성 */
function checkFirstName() {
    var namePattern = /[a-zA-Z가-힣]/;
    if (FirstName.value === "") {
        error[5].innerHTML = "필수 정보입니다.";
        error[5].style.display = "block";
    } else if (!namePattern.test(FirstName.value)) {
        error[5].innerHTML = "한글과 영문 대 소문자를 사용하세요. (특수기호, 공백 사용 불가)";
        error[5].style.display = "block";
    } else {
        error[5].style.display = "none";
    }
}