(window.webpackJsonp=window.webpackJsonp||[]).push([[208],{ehWl:function(e,t,n){"use strict";n.r(t),n.d(t,"formatNumber",(function(){return B})),n.d(t,"ListDetail",(function(){return z})),n.d(t,"styles",(function(){return G}));n("cI1W"),n("PN9k"),n("yIC7");var r=n("1Pcy"),o=n.n(r),i=n("W/Kd"),a=n.n(i),s=n("KEM+"),l=n.n(s),c=n("ERkP"),d=n("iPch"),u=n("rxPX"),m=n("0KEI"),h=n("kHBp"),p=n("RqPI"),f=n("G6rE"),w=function(e,t){var n=t.listId;return n?h.a.select(e,n):void 0},b=function(e,t){var n=w(e,t);return n?f.e.select(e,n.user):void 0},g=function(e,t){return d.h(e,t.listId)},v=Object(u.a)().propsFromState((function(){return{list:w,user:b,loggedInUserId:p.g,media:g}})).propsFromActions((function(){return{createLocalApiErrorHandler:Object(m.d)("LIST_DETAIL"),subscribe:h.a.subscribe,unsubscribe:h.a.unsubscribe}})).withAnalytics(),_=n("aHKU"),y=n.n(_),E=n("wrlS"),C=n("f5/l"),S=n("rHAS"),k=n.n(S),N=n("YeIG"),F=n("Tp1h"),A=n("Jkc4"),I=n("3xO4"),L=n.n(I),U=n("MtXG"),P=n("TIdA"),x=n("A91F"),T=n("rHpw"),H=n("9Xij"),O=n("t62R"),W=n("jV+4"),R=n("/yvb"),V=n("CGyZ"),j=n("3XMw"),M=n.n(j),B=M.a.d58baa7e,D=function(e,t){return c.createElement(M.a.I18NFormatMessage,{$i18n:"j6c94d93"},c.createElement(U.a.Value,null,M.a.d1e8c189({formattedCount:t})),c.createElement(U.a.Label,null,M.a.c94a69ff({count:e})))},X=M.a.e1ad37a2,K=M.a.c84aee16,Y=E.a.getStringValue("home_timeline_spheres_copy_variant");"list_subscribe"===Y&&(D=function(e,t){return c.createElement(M.a.I18NFormatMessage,{$i18n:"be437961"},c.createElement(U.a.Value,null,M.a.bbb96b4c({formattedCount:t})),c.createElement(U.a.Label,null,M.a.hade20a9({count:e})))}),"channel_follow"===Y&&(X=M.a.b12503bd,K=M.a.ac3e20af);var z=function(e){function t(){for(var t,n=arguments.length,r=new Array(n),i=0;i<n;i++)r[i]=arguments[i];return t=e.call.apply(e,[this].concat(r))||this,l()(o()(t),"_renderImage",(function(){var e=t.props.media.image;if(e&&!Object(N.a)(e)){var n=e.url;return c.createElement(c.Fragment,null,c.createElement(y.a,{extend:!0,meta:{property:{"og:image":n}}}),c.createElement(P.a,{accessibilityLabel:"",aspectMode:x.a.exact(3),backgroundColor:T.a.theme.colors.mediumGray,image:e}))}return c.createElement(H.a,{ratio:3},c.createElement(L.a,{style:G.placeholderImageContainer}))})),l()(o()(t),"_renderListDescription",(function(){var e=t.props,n=e.list,r=e.user;if(n){var o=n.name,i=n.description,a=n.member_count,s=n.subscriber_count||0,l=a||0,d="private"===n.mode;return c.createElement(L.a,{style:G.description},c.createElement(L.a,{style:[G.name,G.text]},c.createElement(O.c,{align:"center",size:"large",weight:"bold"},o.trim()),d?c.createElement(k.a,{"aria-label":K,style:G.iconLock}):null),i?c.createElement(c.Fragment,null,c.createElement(y.a,{extend:!0,meta:{property:{"og:description":i.trim()}}}),c.createElement(O.c,{align:"center",style:G.text},i.trim())):null,r?c.createElement(W.a,{isProtected:r.protected,isVerified:r.verified,name:r.name,onLinkClick:t._handleUserNamePress,profileImageUrl:r.profile_image_url_https,screenName:r.screen_name,style:G.text,withLink:!0}):null,t._renderCount(s,l),c.createElement(L.a,{style:G.buttonContainer},t._renderActionButton()))}})),l()(o()(t),"_renderCount",(function(e,n){var r=t.props.basePath,o=B(e),i=B(n);return c.createElement(U.a.Group,null,c.createElement(U.a,{link:r+"/members",onPress:t._handleMembersCountPress,title:i},c.createElement(M.a.I18NFormatMessage,{$i18n:"g44b1b0a"},c.createElement(U.a.Value,null,M.a.aeb31757({formattedCount:i})),c.createElement(U.a.Label,null,M.a.cab254c7({count:n})))),c.createElement(U.a,{link:r+"/subscribers",onPress:t._handleSubscribersCountPress,title:o},D(e,o)))})),l()(o()(t),"_handleEditPress",(function(){t._scribe({element:"edit",action:"click"})})),l()(o()(t),"_handleUserNamePress",(function(){t._scribe({element:"user",action:"click"})})),l()(o()(t),"_handleMembersCountPress",(function(){t._scribe({element:"list_member",action:"click"})})),l()(o()(t),"_handleSubscribersCountPress",(function(){t._scribe({element:"list_subscribed",action:"click"})})),l()(o()(t),"_handleSubscribeActions",(function(){var e=t.props,n=e.list,r=e.subscribe,o=e.unsubscribe,i=e.createLocalApiErrorHandler;if(n){var a=n.following,s=n.id_str;Object(C.a)(i,a,s,r,o,t._scribe)}})),l()(o()(t),"_renderActionButton",(function(){var e=t.props,n=e.basePath,r=e.list,o=e.loggedInUserId,i=e.user;if(r&&r.user){var a=null==i?void 0:i.blocking;return r.user===o?c.createElement(R.a,{link:n+"/info",onPress:t._handleEditPress,type:"secondary"},X):c.createElement(A.a,{customText:r.name,displayMode:F.a.subscribe,userFullName:i&&i.name},(function(e){return c.createElement(V.a,{disabled:a,isFollowing:!!r.following,onFollow:e(t._handleSubscribeActions),onUnfollow:e(t._handleSubscribeActions),showRelationshipChangeConfirmation:!1,type:"list_subscribe"===Y?"subscribe":"list"})}))}})),l()(o()(t),"_scribe",(function(e){t.props.analytics.scribe(Object.assign({},e))})),t}a()(t,e);var n=t.prototype;return n.componentDidMount=function(){this._scribe({action:"impression"})},n.render=function(){var e=this.props,t=e.list,n=e.withRoundedCorners;return t?c.createElement(L.a,{style:[G.container,n&&G.hoverCard]},this._renderImage(),this._renderListDescription()):null},t}(c.Component),G=T.a.create((function(e){return{container:{borderBottomWidth:e.borderWidths.small,borderBottomColor:e.colors.borderColor,borderStyle:"solid"},hoverCard:{borderRadius:e.borderRadii.large,overflow:"hidden"},buttonContainer:{marginVertical:e.spaces.medium},description:{paddingTop:e.spaces.xSmall,paddingHorizontal:e.spaces.xSmall,alignItems:"center"},name:{flexDirection:"row",justifyContent:"center"},iconLock:{color:e.colors.text,marginHorizontal:e.spaces.xxSmall},text:{width:"100%",marginBottom:e.spaces.xSmall},placeholderImageContainer:{backgroundColor:e.colors.mediumGray,height:"100%"}}})),J=v(z);t.default=J},oyNo:function(e,t,n){"use strict";t.__esModule=!0,t.default=void 0,n("PN9k");var r=a(n("7DT3")),o=a(n("ERkP")),i=a(n("OkZJ"));function a(e){return e&&e.__esModule?e:{default:e}}var s=function(e){return void 0===e&&(e={}),(0,r.default)("svg",Object.assign({},e,{style:[i.default.root,e.style],viewBox:"0 0 24 24"}),o.default.createElement("g",null,o.default.createElement("path",{d:"M12.538 6.478c-.14-.146-.335-.228-.538-.228s-.396.082-.538.228l-9.252 9.53c-.21.217-.27.538-.152.815.117.277.39.458.69.458h18.5c.302 0 .573-.18.69-.457.118-.277.058-.598-.152-.814l-9.248-9.532z"})))};s.metadata={height:24,width:24};var l=s;t.default=l},uvhM:function(e,t,n){"use strict";n.r(t),n.d(t,"UserProfileCardContainer",(function(){return D}));var r=n("1Pcy"),o=n.n(r),i=n("W/Kd"),a=n.n(i),s=n("KEM+"),l=n.n(s),c=n("ERkP"),d=n("XnvM"),u=n("P1r1"),m=n("rxPX"),h=n("0KEI"),p=n("G6rE"),f=Object(p.g)([p.a]),w=function(e,t){return f(e,t.screenName)},b=function(e,t){return p.e.selectByScreenName(e,t.screenName)},g=function(e,t){return p.e.selectIsUserNotFound(e,t.screenName)},v=function(e,t){return p.e.selectIsUserSuspended(e,t.screenName)},_=function(e,t){return p.e.selectIsUserWithheld(e,t.screenName)},y=function(e,t){var n=b(e,t);return null==n?void 0:n.profile_interstitial_type},E=function(e,t){var n=b(e,t);return null==n?void 0:n.highlightedLabel},C=function(e,t){var n=function(e,t){return p.e.selectIdByScreenName(e,t.screenName)}(e,t);return{avatarUrls:d.c(e,n),count:d.e(e,n),names:d.f(e,n)}},S=Object(m.a)().propsFromState((function(){return{displaySensitiveMedia:u.j,fetchStatus:w,isNotFound:g,isSuspended:v,isWithheld:_,knownFollowers:C,user:b,userHighlightedLabel:E,userProfileInterstitialType:y}})).propsFromActions((function(){return{createLocalApiErrorHandler:Object(h.d)("USER_PROFILE_CARD"),fetchOneUserByScreenNameIfNeeded:p.e.fetchOneByScreenNameIfNeeded,fetchKnownFollowersIfNeeded:d.a}})).withAnalytics(),k=n("GOQE"),N=n("kGix"),F=n("v//M"),A=n("uIZp"),I=n("v6aA"),L=n("Jkc4"),U=(n("cI1W"),n("0rpg"),n("yIC7"),n("3xO4")),P=n.n(U),x=n("0PHd"),T=n("oSwX"),H=n("jV+4"),O=n("ir4X"),W=n("pBrB"),R=n("wCd/"),V=n("rHpw"),j=function(e){function t(){for(var t,n=arguments.length,r=new Array(n),i=0;i<n;i++)r[i]=arguments[i];return t=e.call.apply(e,[this].concat(r))||this,l()(o()(t),"_renderKnownFollowerSocialContext",(function(e){var t=e.isAllowedToViewFollowersYouKnow,n=e.isLoggedIn,r=e.knownFollowers,o=e.userScreenName,i=r.avatarUrls,a=r.count,s=r.names;return t&&n?c.createElement(P.a,{style:[M.marginTopXSmall,M.minHeight]},"number"==typeof a&&c.createElement(x.a,{knownFollowersAvatarUrls:i,knownFollowersCount:a,knownFollowersNames:s,userScreenName:o})):null})),t}a()(t,e);var n=t.prototype;return n.render=function(){var e=this.props.user;return e?this._renderProfile(e):this._renderEmptyProfile()},n._renderEmptyProfile=function(){var e=this.props,t=e.onAvatarClick,n=e.promotedContent,r=e.screenName;return c.createElement(P.a,{style:M.root},c.createElement(T.default,{accessibilityLabel:r,importantForAccessibility:"no-hide-descendants",onClick:t,promotedContent:n,screenName:r,size:"jumbo",withLink:!0}),c.createElement(P.a,{style:M.marginTopXXSmall},c.createElement(H.a,{screenName:r,withLink:!0,withScreenName:!1,withStackedLayout:!0})))},n._renderProfile=function(e){var t=this.props,n=t.isAllowedToViewOptions,r=t.isLoggedIn,o=t.isUserStatsWithLink,i=t.knownFollowers,a=e.id_str,s=e.name;return c.createElement(P.a,{style:M.root},this._renderUserAvatar({isAllowedToViewFollowButton:n.followButton,isAllowedToViewFullName:n.fullName,isLoggedIn:r,userName:s,userId:a,userAvatarUri:n.avatar?e.profile_image_url_https:void 0}),c.createElement(P.a,{style:M.marginTopXXSmall},this._renderUserName({isAllowedToViewFullName:n.fullName,isProtected:n.badges?e.protected:void 0,isVerified:n.badges?e.verified:void 0,translatorType:n.badges?e.translator_type:void 0,userName:s,withFollowsYou:n.followIndicator&&e.followed_by}),this._renderHighlightedUserLabel({isAllowedToViewLabel:n.label})),this._renderUserDescription({isAllowedToViewDescription:n.description,userDescription:e.description,userEntities:e.entities,userId:a}),this._renderUserStats({isAllowedToViewStats:n.stats,followersCount:e.followers_count,friendsCount:e.friends_count,withLink:o}),this._renderKnownFollowerSocialContext({isAllowedToViewFollowersYouKnow:n.followersYouKnow,isLoggedIn:r,knownFollowers:i,userScreenName:e.screen_name}))},n._renderUserAvatar=function(e){var t=e.isAllowedToViewFollowButton,n=e.isAllowedToViewFullName,r=e.isLoggedIn,o=e.userName,i=(e.userId,e.userAvatarUri),a=this.props,s=a.onAvatarClick,l=a.promotedContent,d=a.followUserButton,u=a.screenName;return c.createElement(P.a,{style:M.row},c.createElement(T.default,{accessibilityLabel:n?o:u,importantForAccessibility:"no-hide-descendants",onClick:s,promotedContent:l,screenName:u,size:"jumbo",uri:i,withLink:!0}),t&&r&&d?d:null)},n._renderUserName=function(e){var t=e.isAllowedToViewFullName,n=e.isProtected,r=e.isVerified,o=e.translatorType,i=e.userName,a=e.withFollowsYou,s=this.props,l=s.onScreenNameClick,d=s.promotedContent,u=s.screenName;return c.createElement(H.a,{badgeContext:"account",isProtected:n,isVerified:r,name:t?i:u,onLinkClick:l,promotedContent:d,screenName:u,translatorType:o,withFollowsYou:a,withLink:!0,withStackedLayout:!0})},n._renderHighlightedUserLabel=function(e){var t=e.isAllowedToViewLabel,n=this.props,r=n.onHighlightedLabelClick,o=n.userHighlightedLabel;if(!t||!o)return null;var i=o.badgeUrl,a=o.description,s=o.link;return c.createElement(O.a,{badgeUrl:i,description:a,link:s,onClick:r,style:M.marginTopXXSmall})},n._renderUserDescription=function(e){var t=e.isAllowedToViewDescription,n=e.userDescription,r=e.userEntities,o=e.userId;return t&&n?c.createElement(P.a,{style:[M.row,M.marginTopXSmall]},c.createElement(W.a,{description:n,entities:r,userId:o})):null},n._renderUserStats=function(e){var t=e.isAllowedToViewStats,n=e.followersCount,r=e.friendsCount,o=e.withLink,i=this.props,a=i.screenName,s=i.onUserStatsPress;return t?c.createElement(P.a,{style:[M.row,M.marginTopXSmall]},c.createElement(R.a,{followersCount:n,friendsCount:r,onPress:s,screenName:a,withLink:o})):null},t}(c.PureComponent),M=V.a.create((function(e){return{root:{padding:e.spaces.small},row:{flexDirection:"row",justifyContent:"space-between"},marginTopXSmall:{marginTop:e.spaces.xSmall},marginTopXXSmall:{marginTop:e.spaces.xxSmall},minHeight:{minHeight:e.spaces.medium}}})),B=n("7wqI"),D=function(e){function t(){for(var t,n=arguments.length,r=new Array(n),i=0;i<n;i++)r[i]=arguments[i];return t=e.call.apply(e,[this].concat(r))||this,l()(o()(t),"_renderUserProfileCard",(function(){var e=t.props,n=e.displaySensitiveMedia,r=e.isNotFound,o=e.isSuspended,i=e.isWithheld,a=e.knownFollowers,s=e.onAvatarClick,l=e.onScreenNameClick,d=e.promotedContent,u=e.screenName,m=e.user,h=e.userHighlightedLabel,p=e.userProfileInterstitialType;if(!m)return null;var f=t.context.loggedInUserId,w=Object(B.a)({displaySensitiveMedia:n,isNotFound:r,isSuspended:o,isWithheld:i,loggedInUserId:f,user:m,userProfileInterstitialType:p}),b=f===m.id_str,g=Object(B.b)({isOwnProfile:b,user:m});return c.createElement(L.a,null,(function(e){return c.createElement(j,{followUserButton:t._renderFollowUserButton(),isAllowedToViewOptions:w,isLoggedIn:!!f,isUserStatsWithLink:g,knownFollowers:a,onAvatarClick:s,onHighlightedLabelClick:t._handleHighlightedUserLabelClick,onScreenNameClick:l,onUserStatsPress:e(),promotedContent:d,screenName:u,user:m,userHighlightedLabel:h,userProfileInterstitialType:p})}))})),l()(o()(t),"_renderFollowUserButton",(function(){var e=t.props,n=e.promotedContent,r=e.showRelationshipChangeConfirmation,o=e.user,i=null==o?void 0:o.id_str;return i?c.createElement(A.a,{promotedContent:n,showRelationshipChangeConfirmation:r,userId:i}):void 0})),l()(o()(t),"_handleFetchUser",(function(){var e=t.props,n=e.createLocalApiErrorHandler;(0,e.fetchOneUserByScreenNameIfNeeded)(e.screenName).catch((function(e){n(k.a)(e)}))})),l()(o()(t),"_handleFetchKnownFollowers",(function(){var e=t.props,n=e.createLocalApiErrorHandler,r=e.fetchKnownFollowersIfNeeded,o=e.user,i=null==o?void 0:o.id_str;i&&r(i).catch(n({}))})),l()(o()(t),"_handleHighlightedUserLabelClick",(function(){var e=t.props,n=e.analytics,r=e.userHighlightedLabel;n.scribe({element:"highlighted_user_label",action:"click",data:{url:null==r?void 0:r.url}})})),t}a()(t,e);var n=t.prototype;return n.componentDidMount=function(){this._handleFetchUser(),this._handleFetchKnownFollowers()},n.componentDidUpdate=function(e){var t,n;(null===(t=e.user)||void 0===t?void 0:t.id_str)!==(null===(n=this.props.user)||void 0===n?void 0:n.id_str)&&this._handleFetchKnownFollowers()},n.render=function(){var e=this.props,t=e.isSuspended,n=e.fetchStatus;return c.createElement(F.a,{fetchStatus:t?N.a.LOADED:n,onRequestRetry:this._handleFetchUser,render:this._renderUserProfileCard})},t}(c.Component);l()(D,"contextType",I.a);var X=S(D);t.default=X},"z2a+":function(e,t,n){"use strict";n.r(t),n.d(t,"default",(function(){return P}));n("PN9k"),n("cI1W");var r=n("97Jx"),o=n.n(r),i=(n("KYm4"),n("1Pcy")),a=n.n(i),s=n("W/Kd"),l=n.n(s),c=n("KEM+"),d=n.n(c),u=n("ERkP"),m=n("zfvc"),h=n("jHwr"),p=n("VY6S"),f=n("w9LO"),w=n("TCjc"),b=n("oyNo"),g=n.n(b),v=n("Oe3h"),_=n("0FVZ"),y=n("7nmT"),E=n.n(y),C=n("rHpw"),S=function(e){var t=e.anchorHeight,n=e.anchorY,r=e.contentHeight;return{canOrientDown:e.viewportHeight-(n+t)>=r+10+15,canOrientUp:n>=r+10+15}},k=function(e){var t=e.anchorWidth,n=e.anchorX,r=e.contentWidth,o=e.viewportWidth,i=r/2,a=n+t/2;return{canOrientCenter:o-a>=i&&a>=i,canOrientStart:n+t>=r,canOrientEnd:o-n>=r}},N=n("jdj2"),F=n.n(N),A=n("/uF9"),I=n.n(A),L=n("3xO4"),U=n.n(L),P=function(e){function t(t,n){var r;return r=e.call(this,t,n)||this,d()(a()(r),"_getBorderRadius",(function(){return C.a.theme.borderRadii.large})),d()(a()(r),"_setContentNode",(function(e){var t=E.a.findDOMNode(e);r._contentNode=t&&t instanceof window.HTMLElement?t:void 0,r._scheduleUpdate()})),d()(a()(r),"_handleEsc",(function(e){var t=r.props.onMaskClick,n=e.altKey,o=e.ctrlKey,i=e.metaKey,a=e.key;!(n||o||i)&&"Escape"===a&&t&&t()})),d()(a()(r),"_updatePosition",(function(){var e=r.props,t=e.anchorNode,n=e.preferredVerticalOrientation,o=e.withArrow,i=e.withFixedPosition;if(r._mounted&&t&&r._contentNode){var a=r._contentNode.scrollHeight,s=r._contentNode.scrollWidth,l=parseFloat(r._getBorderRadius()),c=F.a.get("window"),d=c.height,u=c.width,m=t.getBoundingClientRect(),h=m.left,p=m.top,f=m.height,w=m.width,b=function(e){var t=e.anchorHeight,n=e.anchorY,r=e.contentHeight,o=e.preferredVerticalOrientation,i=void 0===o?"down":o,a=e.viewportHeight,s=e.withFixedPosition,l=S({anchorHeight:t,anchorY:n,contentHeight:r,viewportHeight:a}),c=l.canOrientDown,d=l.canOrientUp,u=window.scrollY;return!d||"up"!==i&&c?{top:s?n+t+10:u+n+t+10}:{bottom:s?a-n+10:-(u+n-10)}}({anchorHeight:f,anchorY:p,contentHeight:a,preferredVerticalOrientation:n,viewportHeight:d,withFixedPosition:i}),g=function(e){var t=e.anchorWidth,n=e.anchorX,r=e.contentWidth,o=e.viewportWidth,i=e.borderRadius,a=r/2,s=t/2,l=n+s,c=k({anchorWidth:t,anchorX:n,contentWidth:r,viewportWidth:o}),d=c.canOrientCenter,u=c.canOrientStart,m=c.canOrientEnd;return!d&&m?i>=s?n-t:n:!d&&u?i>=s?n+2*t-r:n+t-r:d?l-a:0}({anchorWidth:w,anchorX:h,contentWidth:s,viewportWidth:u,borderRadius:l}),v=o?function(e){var t=e.contentStart,n=e.anchorWidth,r=e.anchorX,o=e.contentWidth,i=e.borderRadius,a=n/2,s=r+a,l=o/2;return t+n===r&&i>=a?n+a:t===r?a:t-n==r+n-o?o-n-a:t===r+n-o?o-a:t===s-l?l:s}({contentStart:g,anchorWidth:w,anchorX:h,contentWidth:s,borderRadius:l}):void 0;r.setState({arrowPositionStart:v,bottom:b.bottom,top:b.top,positionStart:g})}})),r.state=Object.freeze({}),r._scheduleUpdate=Object(h.a)(r._updatePosition,window.requestAnimationFrame),r._scheduleDebouncedUpdate=Object(p.a)(r._scheduleUpdate,250),r}l()(t,e);var n=t.prototype;return n.componentDidMount=function(){this._mounted=!0,F.a.addEventListener("change",this._scheduleDebouncedUpdate)},n.componentWillUnmount=function(){var e=this.props.onHoverCardUnmount;this._mounted=!1,F.a.removeEventListener("change",this._scheduleDebouncedUpdate),e&&e()},n.render=function(){var e=this,t=this.props,n=t.children,r=t.onAnimateComplete,i=t.onMaskClick,a=t.show,s=t.withArrow,l=t.withFixedPosition,c=t.withFocusContainer,d=this.state,h=d.bottom,p=d.top,b=d.positionStart,y=void 0===p&&void 0===h,E=y||!c?u.Fragment:f.a,C=I.a.isRTL,S={top:p,bottom:h,left:C?void 0:b,right:C?b:void 0},k=[y?x.initialRenderWrapper:l?x.contentWrapperFixed:x.contentWrapperAbsolute,S],N={borderRadius:this._getBorderRadius()};return u.createElement(_.a.Dropdown,null,u.createElement(w.a.Provider,{value:{isInHoverCard:!0}},i?u.createElement(U.a,{onClick:i,style:x.mask}):null,u.createElement(U.a,{onKeyUp:this._handleEsc,ref:this._setContentNode,style:k},u.createElement(E,null,u.createElement(m.b,{animateMount:!0,duration:"long",onAnimateComplete:r,show:a,type:"fade"},(function(t){var r=t.isAnimating;return u.createElement(v.a,{disableReporting:r},(function(t,r){return u.createElement(U.a,o()({ref:t()},r({style:[x.contentRoot,N]})),s&&u.createElement(g.a,{style:e._getArrowStyle()}),n)}))}))))))},n._getArrowStyle=function(){var e,t=this.state,n=t.arrowPositionStart,r=t.bottom,o=I.a.isRTL;return n?[x.arrow,(e={},e[o?"right":"left"]="calc("+n+"px - "+g.a.metadata.width/2+"px)",e),r?x.downArrow:x.upArrow]:void 0},t}(u.Component),x=C.a.create((function(e){return{arrow:{color:e.colors.cellBackground,filter:"drop-shadow("+e.spaces.nano+" -"+e.spaces.nano+" "+e.spaces.nano+" "+e.colors.lightGray+")",fontSize:e.fontSizes.small,position:"absolute",width:g.a.metadata.width+"px"},contentWrapperAbsolute:{position:"absolute"},contentWrapperFixed:{backfaceVisibility:"hidden",position:"fixed"},initialRenderWrapper:{opacity:0,position:"fixed"},contentRoot:{backgroundColor:e.colors.cellBackground,borderRadius:e.borderRadii.large,boxShadow:e.boxShadows.medium},downArrow:{bottom:"-"+e.fontSizes.xSmall,transform:"rotate(180deg)"},mask:Object.assign({},C.a.absoluteFillObject,{position:"fixed"}),upArrow:{top:"-"+e.fontSizes.xSmall}}}))}}]);
//# sourceMappingURL=https://ton.twitter.com/responsive-web-internal/sourcemaps/web/ondemand.HoverCard.c6562d24.js.map