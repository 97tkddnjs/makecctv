var pw1 = document.querySelector('#password');
var userName = document.querySelector('#username');
var error = document.querySelectorAll('.error_next_box');

/*이벤트 핸들러 연결*/

pw1.addEventListener("focusout", checkPw);
userName.addEventListener("focusout", checkName);

/*콜백 함수*/

/* ID */
function checkName() {
    if (userName.value === "") {
        error[0].innerHTML = "아이디를 입력하세요!";
        error[0].style.display = "block";
    } else {
        error[0].style.display = "none";
    }
}

/* 비밀번호 */
function checkPw() {
    if (pw1.value === "") {
        error[1].innerHTML = "필수 정보입니다.";
        error[1].style.display = "block";
    } else {
        error[1].style.display = "none";
    }
}