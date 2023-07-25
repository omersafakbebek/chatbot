css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAoHCBIVFRgWFRIYGBgYGBgaGRgYGBgYGBgaGRgZGhgYGBgcIS4lHB4rHxgaJzgmKy8xNTU1GiQ7QDszPy80NTEBDAwMEA8QGhISGjQhISE0NDExNDQ0NDQxNDQ0NDQ0NDE0NDQ0NDQ0NDQ0ND80NDQ0NDQ0Pz80NDExNDQxNDQxNP/AABEIAMIBAwMBIgACEQEDEQH/xAAcAAAABwEBAAAAAAAAAAAAAAAAAQIDBAUGBwj/xABAEAACAQIEAwUFBQgABgMBAAABAgADEQQSITEFQVEGImFxgRORobHBBzJCUvAUFiNictHh8SQzc4KSshc0whX/xAAZAQEBAQEBAQAAAAAAAAAAAAAAAQIDBAX/xAAhEQEBAQEAAgMBAAMBAAAAAAAAARECEjEDIUFRBDNhMv/aAAwDAQACEQMRAD8Az2UQFYd4J0YwjL4QZYck4PBVKjBUUmQRgItKZbYXl7W7OVAtwD4zUdn+zdPJcjvHeXEc6ZLGxEKwm87S9mkUBluCBMsvCHte0YK20FobKVJB3ETeAeUQ8sTmhhoB2igI29VV3Nvn7o1+1E/dX1P9hJq4lERDCNIlRtTmte2nd/zJIw/8vqbtbyzR5RZzTBtCtHFfTVNbW1AtceO4ia+xYKF1tluTsNdzpqJPKGU3aFaEjgxV5UwLQWh3h3gwVvCJIioRgJiSIuFARlgyxdoLQG7QWirQrQ0KCHaFAO8KFBAkXgvG80GaA5ebzsPTRtRbTec/vNF2Zr1EuVvrLErq+KoLkNrXtK/BcRRGyGwImbx/aZguWzA+Ux+JxtdnLLmF+e0I6jj8dTc5NzJuH4ehUEqJyXB46sjqzX057++bjA9qUyWZrHkOZlRT9s+HohLWA+ExN5pe1vEmqjwBvMrmkqw5mkbFYwJoPvfKCtVyqT0EojUJNzuZm1ZFjRzOwG5vvv6ma/A4OjRTM4zudr/dHjbmb/ozG8OrZTfp8P8AEtGxxPOZtanLSNjgb2VR6CEmNHQekzyYoneGaxHWc/J0nLULTVuQ18Ib4UHcD6yo4Xi2va29pqqaLub3/XQ+M1z9lmM1X4Ol7qcp58wfTlK7E4dkOu3IzXYqiuuoHPl8ZU4oKRlYaHrHleU8fJQBoeaFiKZRreoPURsNOkuuVmXDuaDNG80F5ULvDvG7w7wF3gvEZoM0NFwogmDNAMmEYRMSTAVBEZoIDt4LxMKA9h6RdrD18J0TsnhEI1G2kyPC0AXzmg4BiirEDk2s1Ga2ON4RSZblRMNxTCBHyi1jtN4cU2TXpMTie/iUuCQS2sJUIYF7gEaHpNVwzs1TYAsLmHiMIoF+gllwHG5kECh7RcFRUYgbCc2xFIo1uXKdg7TPem3kflOb4rDhl1/1JixluJv3D4kSnDS44stlt0b6GUx3mK3D9JpKpVPGRqCayWigfraZaiTTvsAT5SbSpNzT3kyD+1N91LDxgbF1U/FcevynOusaTA4QAgkMviDcDzmiw9N7b33mM4V2hqIbMAwmuwnaCky91ADvbcX/AF8pebnsu30kVkfILAG29z0v/f4yhxruT933W6+EucTxN2sqKup15n4H09JDxVMDXMdfLw0sI6sXmYoMcLqDzB+B/QlfeW2PQ5TKia+K7HL5pnRV4LxMVOrmO8O8TBAVeETCggC8F4mEYCiYkmEYV4Cs0KJvBAkQQQQLXhVS4K9JpOz9As+n5hMrwkd4+U2vZE/xCPKajNa9sGcvptM++Fy1FJGlzNwAMszvEhd7CZvcntvn4+uvSHj8Qo0jvAMK2UHziKtBtL7TQ8NVcotHnKvXw9czaznadGyEciCDMBUey68hOpdpKYyN5GcpxYujDwM1+Of6zeOXOCMyhibhSbE2OthKMixIIsbmSeI3NXoFQH4n6mIR85623M42usn0k4JDYmFXzcpKokKo8genKO0ACdQJKvPtUGqVG8nYDEo7imVrM7EKoQU9WOws395c/wD81XXLa3l8zHeH4WthnzoUDAWDZRmA8yJmWN2Vm8WpViNQVJGxU3G4ZTqrDprJHD+Iup6zSVOGiozVatndh3raBhsLgdORErsJgEVx3QQDt/uTrIvOg3ErDXT9bSK/FibBSb++WXHuGK1UopygkWPmLgfSUY4NXRwFNMaDMzimwBvY9176c9BHMlOrYsqWMqgWfVTof1yjIMQ6MlUoDnQE98AhWFr6A7HlppFzpxPbn8l3BwXhQ50cwgEEEBUSYqJMAoUBgMAiYkmKMQYAghQQJcEBhQLfhSd0nqZpOyVT/iCPAfOZ3hn3P11lx2Sf/ij/AEj6zUZrrX4fSZnHIRUzTSL930ldVw4ZtZw+Sa9n+L3ObbVLWrgggGXfBSQgvIVDgoRma978pZ4Ncukxxz1Ltej5++Ouc5R+0g/ht5TkbNe/mZ1/jwvTbynHaIuWHRmHxnpnp8z9Zni2H7xFtToPiQPjIGDQqctj43622mi4rS121tp5j/coqD9/LZrgXN+VuQ985dT7dOb9FOTbyESlciCu2sigyVeWl4ZjCLAnSaqnUVwCbX623P6BnO8PWsd5puD1ixK3IvYjcb8rjaSN2rTieLVFyjVja1t/InYWI38JU4VWz94G1xI3aJaiOjU1JUJuAT3ybkn4cuUp349XJtV22Btl+e+0x1zbWuepPbofFuGPlD2P3QVvzHK3XYyoroH3AJt6+fylLT7TC2rsbAW1LE22FvSWWGx61FDWtfXUEXPMjpJJY3cs9ltRQIQLc+nS0qJKxGIOawMizr8X64fLn0EOFDnVyCCCCAIRhxJgAwocSTABiDFGJMBMEEEgmQQQSi24U/dtLns4MuKU9V+R/wAzN4Ctla3IzU9nwDWUzUSur0PuekrcXiMh16yZRey+khYinn8Zjx1rnrxB8ett/SSMA+bWV74UchrJGFYpvyict9fJMyJHGR3D5TjdJLO/9bTrvEat1I6icr4qVpNUY7KSfPw981PTj+qbip7w9fpKFAMznw+v+JCxOJqEklib9dfd/aBcT47zHTcKqm8aAhO8QHmWofQy44disvh+tpRFpMwhLaA200mbG5caWjjlc2JFuZJAllha1GxBZfDmNhubW6zE4bAuxI9pqOX+PrLihw2uqFkqo1jbIbqx/pvoSPEiJMW3fxe4zB0WAGSnfQ5RkuQRvYbyDjiAumltLeQtb3CRsTQxKckYAX7r3O221r+RkMYwuCW0PO/UaSdRZc+qIG5JhwlFoYnXmZHHu7Rw4m8F5pkcEKCAcIwQGAUTDMTABMSYcSZAUEEECbChwpQcuuCcUyOpblzlJeHCV2vhvGKbqCGB0icZxFUUm85FgOJ1KRurG3SXJ4uaw1b0mpUa09qaZNswvLfA49XW/WczHD1Jz8+nWT6PFzRXfSBu+K45EXe05L2s4gHDhT95hfyGv0jnFOOVKp3sJnuIvoBfxktWRWtTvqDI7q2xj4e0UHB/DcjlM4qEKhG8WHi2ZG5WkZkI21EzYupCtJVFDuJXo8mYfEWma3LEyqjmzaqw56++8mUON100IV/Fl73qwsT6xzBcQI2t7ryU2MzizKp6aWt7pNbz+Uh+L1KgsVVRzC31950kRNSTCxlZEF/1fpG8BiM6m4sQfgdv14S8zftnrrEqHeJvBedHIq8F4m8OUHeCFBAOEYLxJMAExMF4UgEIw4kwBBBBAlZoLzp5+z6j0PvMbb7P6Pj75rE1zW8F50r/AOPKfU+8xB+zyn1b3xhrm94dOoVNxOin7PE/M3viG+zxfzN74w1jl4pp4yvxFcubn/U2GO7EZLnPlVRdmYgADqSdhMtjaVEG1MswG7HQH+kb28/dH2bEO8qsfUufIWls56CQMThwwI639/L42ksNVee8BNiDsfgYwpPMeskU+h1EihWp5u8u/MSO6kbiSxTKm41Xpz9IVVNNNj7x4RghHWAORFMkKQLp4lhteTkr1CBfQE6E69Lj4yvGsvuJY2lVKqgcIl/Z5woYZrdw5SRYZQFtbnpc6zF3FRiKhLAG+n9+km8MUo3esMw23OmoPhz98ZckscqkHmTsOd5oOzHZqpii7ofuFQSdizA6AeAHxE1Izpi8F5qP3HxH5h7oP3GxP5h7pcpsZi8E037k4n8w90H7k4nqPcZMpsZmC80p7FYrqPcYk9i8V1HuMZTYzd4RM0h7F4rw+Mit2WxOa1vXWMpsUcE0X7l4rw+MH7mYrw+MZTYzkIzSfubiv5YR7GYrw+MZTYzkE0P7m4vwglymx3Q4lY02JAjf7MYT4YmdJzy43rpLWsLRaODIS0yLGOUhrJeZ+NTqpwAldxji1DDUy9Vwo5LuzHoq8z+jInaPtDSwdPM3edh3KYOrHqeijmfrON8X4rVxFQ1Kr5mO3JVH5VHIfo3Os5ya6W4n9pu01XFtr3KYPdpg6eDOfxN8By6nOs0UxiGM2wQTrG6qxTbXhtqIVXGkM7KRcNYyJWpmkRfvITpfl4GWGIHeB8PlE4qlnS3Qg/T6yWLKYQAjTn6g/WGEPMeY8pApuUJVhcX1H1En0XOmtx15+Ukq0ipS/kPvkc0l2IKnqdpZ6xLoDuLeMuGq6pQtYbx72SaG5HlHKtOxFtdPqLx5ABy1kxCGxBY63152t8p0X7Me0dDDoaFZQi1HLiryDEBbP0FlFjy59Zz5gSDfa21tfUyRQqa2G4+IlWvSNNENiLEHUEag+Rjvs0nCuB9pMThv+XUJTmjd5Pd+H0tOi8E7Z0MRZXPsqh0yue6x/lfb0NjNZv653qz8atkTwji00kQoxjiBvSW8/XtJ1f4kCinSK/Zl6RmiTzkxZz6+nTm6YbCr0kF8El9hLYyG+8kq2FjCrbaEcKvSSBtCaNVXtQUHYRz2KeELEIeUYfNedZzv643qw97JPCFI2VukOa8P+p53+LA1YnMYytbwjgxHhOfp0ygUJlfxvilPCUjUfU7It7Fm5AeHMnkI7xXjVPDU2q1TZV6bsx+6ijmx/wA7AzjHaHtBUxVUu+g2RAe6i9B49TzPoA2+jxhvifEKleo1So2Z2PoByVRyUchILNIzVTeJaodxAfdolW3HSRauKynvDun8Q2HmI+SLgjYj/UGD2067QKdxBv5iE3WAzihoD4/OBB8RF4hbr6iIVtv17oDGIoq4OnL1EiI2QhW+6fuuPrLYLIppi5RhodR4SWLKJbjxjquDykOk5ptkfUfhMnlOYgN1hoD429+nztAscKXUgcxp4RNA3H65wAiMPxfCwhYmlaxXlHcvXWOZbgiUIoVbi/PmI+r+6QspDXGht744lXw05gf2hGu4F2vxWGsob2lP8jk2H9Lbr8vCdL4D2jw+LXuNlcfepvYOPL8w8ROG0qgOgN/n5EcjHVcghlJBGoI0IPUEbSpY9EUqclCZDsP2oGKo5XP8anYP/OPwuB47Hx8xNR+0TF2t8/SRIVXeOHESFVxBvtJIqzTaHIlPE6bRRxPhGJp17RksIxVxB6RtKpM6SfTFSM0EbzwS7Eyl4RUYaST7BZT8IfKAp5Sq+0ftR+xYayN/HrXSntdBbv1bfyg2H8zL4znXSOaduO0DYnEuAf4VJ2Sko2NjlaoepYg2PJbdTfNe3swHWRkNhaN4hufSDE+qe9DUyOawbc7gFT8x744WlQ69MMpB2MhYNyrGmx8UMnI0h8Qp6Bx95Tf05xSJjvsR6x5CDI1Fw6g9RrCpOQSJQ8w7pEaK2+kdzbxsm48jb3QHViKyc+kUu0UohETEUg6eI2MRga/4W3ElWsZDxdIqcwkVYgRhRlYjlf56j43HpF4ermWFXGmbpv5c/dofSA6qxSkDeNX0BHqIDKg3S+o3HyjNQWIYesk0zcjUC+hJvYeJtraaduzNOouSk5d/aVVzEBbhVBprlZh3W3DrcXbpqIrJOAbG9j16efUeEkIx2Oh+HmPCL4jw72JIFRXQu6Iw7rXpkBgyHVSMwvvrz6xHfQN00PlKLPh+NqUKi1KbZWU+hHNSOYPSdo4RxiniaC1kNr6Ot9UcfeU/ra04WH0lp2e4w2Hqak+zfu1F8OTDxH95ZmzUuyXHZVxQP4hA1UdZX4HhxsDe4OoN9CDqCJLfBm81138cuOfN6s2nf2gfmgTEAm2aMfsJtaMJw9g4OsTr46XrqLp6V1veN0aVidZJVe5GFGsx7rr+EXHWCR6mHNzvBO3jz/XPy6Q14mg1uJyD7RuK/tGOaxutNUpL07oLuf8AzZh/2yciVfzMPUzD58zFzuxJ/wDI3PzmPm+Pwz7b5unFMDC4tEodTFNOLZul+U8tV+oktHuJGbXUbiLqPl7w2PwPOWCYhjtQaSGlTY8pMRgRveVmoOCbIzIeRuPIyVVGoMg44FWVxy0P0kpK4YRP4HgYhdGI6wkMn8J4ecRXSmHy3DtfJ7Q2RGchae7sQpATmZTGs+z3gDVW9vdM1Pv0qNVDkrAaFw5FsgJy5lDFWykjYNe9q+zSYtWxWGplay39vRIsxZbZgVG1UCx00YEEE3BNnhKmIZqDO5qDIGpKmHXD+wDUmyvXVnLojMpWw7pykG+kZfGENUZscUbuFsQrYRBlDuUwxQqWByOjZ2VrAkggEg437axyV1uLxplzCbTj3A6XsnxCV71e69Skz4drB7ZgDRAUuHLaiwOU2HOY4jXzm/bPpX0GKPbkZYfWQscn4hykjD1Mygyf8KVhzup5fT/Fj6xUavlfz/0f/wA+6O7RASmxm8UYavgqedXepSpsQgqZUco9QFDcEh1Sx7o1DpfSxGCqDnNJ2R4gqM+YZkVfaMtyCUQgVgttb+yZ3sNzSSKsX9Ph2HxWGqipVqCoipUpXdaqogTMKiMtNGdSoYMrAtdGNiwvMBjcHUpsyOu4zCxDK6tezo40dDY2YfSdZo8OSjiVbN/CUFEVVUZkc50YMNGCZm2H42vckk4rtlwr2LgLth/4JA/DTZ3q4Zv6Sjsl/wA1JuomebrfXF5kt/WRwtYWCkm/In8Q6jrH8/KQ8M9i6MLgG48L6j5xxntudtfMTTLs/wBnvFjVwwQm70Tk8ch1Q+64/wC2anMZyj7L8aExmRjYVUdbdWXvr62Vp2IoszeYzOdRMxhBzJRCwALGcr4otau2WRcNi2J1lnURbGQ0oLeahge2PWCOtREKTV8Y5jXH8J/+m/8A6mcsp7D9chDgno/yfcY4/R0eccq7CCCeeemzQ3jp+43p8xCgkiU1h9vWS+H7v6QQSwo+JfdPp84zhYUEfpExYdVypBUkEFSCNCDm3B5QQSoN6z1FzVGLsRqzEsTr1MNRBBKU8sUYIIQ1W2MjcP2PnCgkqn33Hr8o425hwRADtLLs1/zk8RVB8R7GppCggdL4JrhMATqfZ0Rc9NdPgPdKftgP4p8eGG/jlrgrfy5dIIJyn/qvV8n+rlys/wDPP9IjmI29D9IIJ0jztF2P/wDv4X/qr/6tO8vBBHTXJow1ggmY6UrkY3T3hQSxy69nmggghp//2Q==">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''